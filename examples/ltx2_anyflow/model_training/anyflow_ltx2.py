import torch


class AnyFlowTimeAdapter(torch.nn.Module):
    """Zero-init conditioning adapter for LTX2 flow-map training.

    The base LTX2 DiT only has one timestep input. This adapter injects the
    target endpoint r through the patchified latent tokens without changing the
    original DiffSynth model files. The final projection is zero-initialized, so
    attaching it preserves the pretrained model's behavior at step zero.
    """

    def __init__(self, out_dim, hidden_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, timestep, r_timestep, seq_len, dtype, device):
        t = timestep.to(device=device, dtype=torch.float32).reshape(-1, 1)
        r = r_timestep.to(device=device, dtype=torch.float32).reshape(-1, 1)
        delta = (t - r).clamp(min=0.0)
        features = torch.cat([t, r, delta, delta / (t + 1e-6)], dim=-1).to(dtype=dtype)
        emb = self.net(features).to(dtype=dtype)
        return emb[:, None, :].repeat(1, seq_len, 1)


def attach_anyflow_time_adapters(dit, video_dim=None, audio_dim=None, hidden_dim=256):
    if video_dim is not None and not hasattr(dit, "anyflow_video_time_adapter"):
        dit.anyflow_video_time_adapter = AnyFlowTimeAdapter(video_dim, hidden_dim=hidden_dim)
    if audio_dim is not None and not hasattr(dit, "anyflow_audio_time_adapter"):
        dit.anyflow_audio_time_adapter = AnyFlowTimeAdapter(audio_dim, hidden_dim=hidden_dim)
    return dit


def sample_flowmap_timesteps(pipe, device, dtype, diffusion_ratio=0.25, consistency_ratio=0.25):
    """Sample a scalar t/r pair in the scheduler's 0..1000 timestep scale."""
    t1 = torch.rand((), device=device, dtype=torch.float32)
    t2 = torch.rand((), device=device, dtype=torch.float32)
    t = torch.maximum(t1, t2)
    r = torch.minimum(t1, t2)

    mode = torch.rand((), device=device, dtype=torch.float32)
    if mode < diffusion_ratio:
        r = t
    elif mode < diffusion_ratio + consistency_ratio:
        r = torch.zeros_like(t)

    timestep = (pipe.scheduler.num_train_timesteps * t).reshape(1).to(dtype=dtype)
    r_timestep = (pipe.scheduler.num_train_timesteps * r).reshape(1).to(dtype=dtype)
    return timestep, r_timestep


def scale_noise_at_timestep(pipe, sample, noise, timestep):
    sigma = (timestep.to(device=sample.device, dtype=sample.dtype) / pipe.scheduler.num_train_timesteps)
    sigma = sigma.view(*sigma.shape, *([1] * (sample.ndim - sigma.ndim)))
    return sigma * noise + (1.0 - sigma) * sample


def flowmap_step(pipe, sample, model_output, timestep, r_timestep):
    t = (timestep.to(device=sample.device, dtype=sample.dtype) / pipe.scheduler.num_train_timesteps)
    r = (r_timestep.to(device=sample.device, dtype=sample.dtype) / pipe.scheduler.num_train_timesteps)
    t = t.view(*t.shape, *([1] * (sample.ndim - t.ndim)))
    r = r.view(*r.shape, *([1] * (sample.ndim - r.ndim)))
    return sample - (t - r) * model_output


def FlowMapSFTAudioVideoLoss(
    pipe,
    diffusion_ratio=0.25,
    consistency_ratio=0.25,
    flowmap_reconstruction_weight=0.0,
    **inputs,
):
    timestep, r_timestep = sample_flowmap_timesteps(
        pipe,
        device=pipe.device,
        dtype=pipe.torch_dtype,
        diffusion_ratio=diffusion_ratio,
        consistency_ratio=consistency_ratio,
    )

    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = scale_noise_at_timestep(pipe, inputs["input_latents"], noise, timestep)
    training_target = noise - inputs["input_latents"]

    # audio
    training_target_audio = None
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = scale_noise_at_timestep(pipe, inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = audio_noise - inputs["audio_input_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep, r_timestep=r_timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    if hasattr(pipe.scheduler, "training_weight"):
        loss = loss * pipe.scheduler.training_weight(timestep)

    if training_target_audio is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        if hasattr(pipe.scheduler, "training_weight"):
            loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio

    if flowmap_reconstruction_weight > 0:
        pred_video_r = flowmap_step(pipe, inputs["video_latents"], noise_pred, timestep, r_timestep)
        target_video_r = scale_noise_at_timestep(pipe, inputs["input_latents"], noise, r_timestep)
        loss = loss + flowmap_reconstruction_weight * torch.nn.functional.mse_loss(
            pred_video_r.float(), target_video_r.float()
        )
        if training_target_audio is not None:
            pred_audio_r = flowmap_step(pipe, inputs["audio_latents"], noise_pred_audio, timestep, r_timestep)
            target_audio_r = scale_noise_at_timestep(pipe, inputs["audio_input_latents"], audio_noise, r_timestep)
            loss = loss + flowmap_reconstruction_weight * torch.nn.functional.mse_loss(
                pred_audio_r.float(), target_audio_r.float()
            )
    return loss


def model_fn_ltx2_anyflow(
    dit,
    video_latents=None,
    video_context=None,
    video_positions=None,
    video_patchifier=None,
    audio_latents=None,
    audio_context=None,
    audio_positions=None,
    audio_patchifier=None,
    timestep=None,
    r_timestep=None,
    input_latents_video=None,
    denoise_mask_video=None,
    ref_frames_latents=None,
    ref_frames_positions=None,
    in_context_video_latents=None,
    in_context_video_positions=None,
    input_latents_audio=None,
    denoise_mask_audio=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    timestep = timestep.float() / 1000.0
    r_timestep = timestep if r_timestep is None else r_timestep.float() / 1000.0

    b, c_v, f, h, w = video_latents.shape
    video_latents = video_patchifier.patchify(video_latents)
    seq_len_video = video_latents.shape[1]
    attach_anyflow_time_adapters(dit, video_dim=video_latents.shape[-1])
    video_latents = video_latents + dit.anyflow_video_time_adapter(
        timestep, r_timestep, video_latents.shape[1], video_latents.dtype, video_latents.device
    )
    video_timesteps = timestep.repeat(1, video_latents.shape[1], 1)

    if input_latents_video is not None:
        denoise_mask_video = video_patchifier.patchify(denoise_mask_video)
        video_latents = video_latents * denoise_mask_video + video_patchifier.patchify(input_latents_video) * (1.0 - denoise_mask_video)
        video_timesteps = denoise_mask_video * video_timesteps

    total_ref_latents = ref_frames_latents if ref_frames_latents is not None else []
    total_ref_positions = ref_frames_positions if ref_frames_positions is not None else []
    total_ref_latents += [in_context_video_latents] if in_context_video_latents is not None else []
    total_ref_positions += [in_context_video_positions] if in_context_video_positions is not None else []
    if len(total_ref_latents) > 0:
        for ref_frames_latent, ref_frames_position in zip(total_ref_latents, total_ref_positions):
            ref_frames_latent = video_patchifier.patchify(ref_frames_latent)
            ref_frames_timestep = timestep.repeat(1, ref_frames_latent.shape[1], 1) * 0.0
            video_latents = torch.cat([video_latents, ref_frames_latent], dim=1)
            video_positions = torch.cat([video_positions, ref_frames_position], dim=2)
            video_timesteps = torch.cat([video_timesteps, ref_frames_timestep], dim=1)

    if audio_latents is not None:
        _, c_a, _, mel_bins = audio_latents.shape
        audio_latents = audio_patchifier.patchify(audio_latents)
        attach_anyflow_time_adapters(dit, audio_dim=audio_latents.shape[-1])
        audio_latents = audio_latents + dit.anyflow_audio_time_adapter(
            timestep, r_timestep, audio_latents.shape[1], audio_latents.dtype, audio_latents.device
        )
        audio_timesteps = timestep.repeat(1, audio_latents.shape[1], 1)
    else:
        c_a, mel_bins = None, None
        audio_timesteps = None

    if input_latents_audio is not None:
        denoise_mask_audio = audio_patchifier.patchify(denoise_mask_audio)
        audio_latents = audio_latents * denoise_mask_audio + audio_patchifier.patchify(input_latents_audio) * (1.0 - denoise_mask_audio)
        audio_timesteps = denoise_mask_audio * audio_timesteps

    vx, ax = dit(
        video_latents=video_latents,
        video_positions=video_positions,
        video_context=video_context,
        video_timesteps=video_timesteps,
        audio_latents=audio_latents,
        audio_positions=audio_positions,
        audio_context=audio_context,
        audio_timesteps=audio_timesteps,
        sigma=timestep,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )

    vx = vx[:, :seq_len_video, ...]
    vx = video_patchifier.unpatchify_video(vx, f, h, w)
    ax = audio_patchifier.unpatchify_audio(ax, c_a, mel_bins) if ax is not None else None
    return vx, ax
