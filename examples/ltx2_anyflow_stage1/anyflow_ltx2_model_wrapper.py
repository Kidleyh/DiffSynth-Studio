import copy
from contextlib import contextmanager
from typing import Optional

import torch


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, (list, tuple)) else [value]


class AnyFlowAdaLayerNormSingle(torch.nn.Module):
    """AdaLN wrapper that blends the original t embedding with a copied r embedding."""

    def __init__(self, base_adaln: torch.nn.Module, gate: float = 0.25, timestep_scale_multiplier: float = 1000.0):
        super().__init__()
        self.base_adaln = base_adaln
        self.r_adaln = copy.deepcopy(base_adaln)
        self.gate = torch.nn.Parameter(torch.tensor(float(gate)))
        self.timestep_scale_multiplier = float(timestep_scale_multiplier)
        self._r_timestep = None
        for param in self.base_adaln.parameters():
            param.requires_grad = False
        for param in self.r_adaln.parameters():
            param.requires_grad = True
        for param in self.r_adaln.linear.parameters():
            param.requires_grad = False
        self.frozen_unused_r_adaln_linear = True

    def set_r_timestep(self, r_timestep: Optional[torch.Tensor]):
        self._r_timestep = r_timestep

    def forward(self, timestep: torch.Tensor, hidden_dtype: Optional[torch.dtype] = None):
        if self._r_timestep is None:
            return self.base_adaln(timestep, hidden_dtype=hidden_dtype)
        emb_t = self.base_adaln.emb(timestep, hidden_dtype=hidden_dtype)
        r = self._r_timestep.to(device=timestep.device, dtype=torch.float32).flatten()
        if r.numel() == 1 and timestep.numel() != 1:
            r = r.expand(timestep.numel())
        elif r.numel() != timestep.numel():
            r = r.reshape(-1)[:1].expand(timestep.numel())
        emb_r = self.r_adaln.emb(r * self.timestep_scale_multiplier, hidden_dtype=hidden_dtype)
        gate = self.gate.to(device=emb_t.device, dtype=emb_t.dtype).clamp(0.0, 1.0)
        emb = gate * emb_t + (1.0 - gate) * emb_r.to(dtype=emb_t.dtype)
        return self.base_adaln.linear(self.base_adaln.silu(emb)), emb


class LTX2AnyFlowWrapper(torch.nn.Module):
    """Non-invasive AnyFlow stage-1 wrapper for DiffSynth LTX2 DiT."""

    ADALN_FIELDS = (
        ("adaln_single", "timestep_scale_multiplier"),
        ("audio_adaln_single", "timestep_scale_multiplier"),
        ("prompt_adaln_single", "timestep_scale_multiplier"),
        ("audio_prompt_adaln_single", "timestep_scale_multiplier"),
        ("av_ca_video_scale_shift_adaln_single", "timestep_scale_multiplier"),
        ("av_ca_audio_scale_shift_adaln_single", "timestep_scale_multiplier"),
        ("av_ca_a2v_gate_adaln_single", "av_ca_timestep_scale_multiplier"),
        ("av_ca_v2a_gate_adaln_single", "av_ca_timestep_scale_multiplier"),
    )

    def __init__(self, dit: torch.nn.Module, gate: float = 0.25, freeze_base: bool = True):
        super().__init__()
        self.dit = dit
        self.gate = gate
        if freeze_base:
            for param in self.dit.parameters():
                param.requires_grad = False
        self._install_anyflow_adaln()

    def _install_anyflow_adaln(self):
        found = []
        for name, scale_attr in self.ADALN_FIELDS:
            module = getattr(self.dit, name, None)
            if module is None:
                continue
            if isinstance(module, AnyFlowAdaLayerNormSingle):
                found.append(name)
                continue
            if not hasattr(module, "emb") or not hasattr(module, "linear") or not hasattr(module, "silu"):
                continue
            scale = getattr(self.dit, scale_attr, getattr(self.dit, "timestep_scale_multiplier", 1000))
            setattr(self.dit, name, AnyFlowAdaLayerNormSingle(module, gate=self.gate, timestep_scale_multiplier=scale))
            found.append(name)
        if not found:
            raise RuntimeError(
                "LTX2AnyFlowWrapper could not locate LTX2 timestep embedding fields. "
                "Please confirm ltx2_dit.py still exposes fields such as adaln_single/audio_adaln_single."
            )

    def anyflow_modules(self):
        for module in self.dit.modules():
            if isinstance(module, AnyFlowAdaLayerNormSingle):
                yield module

    @contextmanager
    def r_timestep_context(self, r_timestep):
        for module in self.anyflow_modules():
            module.set_r_timestep(r_timestep)
        try:
            yield
        finally:
            # Keep the latest r_timestep available after forward so PyTorch
            # gradient checkpoint recomputation sees the same AnyFlow r state.
            # Later wrapper calls overwrite it before entering the DiT again.
            pass

    @staticmethod
    def _normalize_time(timestep, batch_size: int, device, dtype):
        timestep = torch.as_tensor(timestep, device=device, dtype=dtype).flatten()
        if timestep.numel() == 1:
            timestep = timestep.expand(batch_size)
        return timestep.clamp(0.0, 1.0).view(batch_size, 1, 1)

    def anyflow_model_fn_ltx2(
        self,
        video_latents,
        audio_latents=None,
        video_context=None,
        audio_context=None,
        video_positions=None,
        audio_positions=None,
        timestep=None,
        r_timestep=None,
        video_patchifier=None,
        audio_patchifier=None,
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
    ):
        if timestep is None or r_timestep is None:
            raise ValueError("LTX2AnyFlowWrapper.forward requires both timestep and r_timestep in normalized [0, 1].")
        if video_context is None or video_positions is None:
            raise ValueError("video_context and video_positions are required for LTX2AnyFlowWrapper.forward.")
        if video_patchifier is None:
            raise ValueError("video_patchifier is required for native-style LTX2 AnyFlow forward.")

        batch_size = video_latents.shape[0]
        dtype = video_latents.dtype
        device = video_latents.device
        t = self._normalize_time(timestep, batch_size, device, dtype)
        r = self._normalize_time(r_timestep, batch_size, device, dtype)

        if video_latents.ndim != 5:
            raise ValueError("Native-style LTX2 AnyFlow forward expects unpatchified 5D video latents.")
        _, _, frames, height, width = video_latents.shape
        video_latents = video_patchifier.patchify(video_latents)
        seq_len_video = video_latents.shape[1]
        video_timesteps = t.repeat(1, seq_len_video, 1)

        if input_latents_video is not None:
            if denoise_mask_video is None:
                raise ValueError("input_latents_video requires denoise_mask_video for native LTX2 conditioning.")
            denoise_mask_video = video_patchifier.patchify(denoise_mask_video)
            video_latents = video_latents * denoise_mask_video + video_patchifier.patchify(input_latents_video) * (1.0 - denoise_mask_video)
            video_timesteps = denoise_mask_video * video_timesteps

        ref_latents = _as_list(ref_frames_latents) + _as_list(in_context_video_latents)
        ref_positions = _as_list(ref_frames_positions) + _as_list(in_context_video_positions)
        if len(ref_latents) != len(ref_positions):
            raise ValueError(
                "Reference/in-context video conditioning has mismatched latents and positions: "
                f"{len(ref_latents)} latents vs {len(ref_positions)} positions."
            )
        for ref_latent, ref_position in zip(ref_latents, ref_positions):
            ref_latent = video_patchifier.patchify(ref_latent)
            ref_timestep = t.repeat(1, ref_latent.shape[1], 1) * 0.0
            video_latents = torch.cat([video_latents, ref_latent], dim=1)
            video_positions = torch.cat([video_positions, ref_position], dim=2)
            video_timesteps = torch.cat([video_timesteps, ref_timestep], dim=1)

        audio_shape = None
        if audio_latents is not None:
            if audio_patchifier is None:
                raise ValueError("audio_patchifier is required when audio_latents are provided.")
            if audio_latents.ndim != 4:
                raise ValueError("Native-style LTX2 AnyFlow forward expects unpatchified 4D audio latents.")
            _, audio_channels, _, mel_bins = audio_latents.shape
            audio_shape = (audio_channels, mel_bins)
            audio_latents = audio_patchifier.patchify(audio_latents)
            audio_timesteps = t.repeat(1, audio_latents.shape[1], 1)
            if input_latents_audio is not None:
                if denoise_mask_audio is None:
                    raise ValueError("input_latents_audio requires denoise_mask_audio for native LTX2 conditioning.")
                denoise_mask_audio = audio_patchifier.patchify(denoise_mask_audio)
                audio_latents = audio_latents * denoise_mask_audio + audio_patchifier.patchify(input_latents_audio) * (1.0 - denoise_mask_audio)
                audio_timesteps = denoise_mask_audio * audio_timesteps
        else:
            audio_timesteps = None

        with self.r_timestep_context(r.view(batch_size)):
            video_out, audio_out = self.dit(
                video_latents=video_latents,
                video_positions=video_positions,
                video_context=video_context,
                video_timesteps=video_timesteps,
                audio_latents=audio_latents,
                audio_positions=audio_positions,
                audio_context=audio_context,
                audio_timesteps=audio_timesteps,
                sigma=t,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        video_out = video_out[:, :seq_len_video]
        video_out = video_patchifier.unpatchify_video(video_out, frames, height, width)
        if audio_out is not None and audio_shape is not None:
            audio_out = audio_patchifier.unpatchify_audio(audio_out, *audio_shape)
        return video_out, audio_out

    def forward(
        self,
        video_latents,
        audio_latents=None,
        video_context=None,
        audio_context=None,
        video_positions=None,
        audio_positions=None,
        timestep=None,
        r_timestep=None,
        video_patchifier=None,
        audio_patchifier=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **kwargs,
    ):
        return self.anyflow_model_fn_ltx2(
            video_latents=video_latents,
            audio_latents=audio_latents,
            video_context=video_context,
            audio_context=audio_context,
            video_positions=video_positions,
            audio_positions=audio_positions,
            timestep=timestep,
            r_timestep=r_timestep,
            video_patchifier=video_patchifier,
            audio_patchifier=audio_patchifier,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **kwargs,
        )


def trainable_state_dict(module: torch.nn.Module):
    return {name: param.detach().cpu() for name, param in module.named_parameters() if param.requires_grad}


def load_trainable_state_dict(module: torch.nn.Module, state_dict, strict_trainable=True):
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    critical_tokens = ("lora", "r_adaln", "r_timestep", "gate", "time_adapter", "anyflow")
    critical_unexpected = [key for key in unexpected if any(token in key for token in critical_tokens)]
    if critical_unexpected:
        print("critical unexpected keys while loading AnyFlow checkpoint:")
        for key in critical_unexpected:
            print(f"  {key}")
        raise RuntimeError(f"Unexpected critical AnyFlow/LoRA keys: {critical_unexpected}")
    if strict_trainable:
        trainable_names = {name for name, param in module.named_parameters() if param.requires_grad}
        missing_trainable = sorted(name for name in trainable_names if name not in state_dict)
        if missing_trainable:
            print("missing trainable keys while loading AnyFlow checkpoint:")
            for key in missing_trainable:
                print(f"  {key}")
            raise RuntimeError("Checkpoint is missing trainable AnyFlow/LoRA weights.")
    frozen_missing = [key for key in missing if key not in {name for name, param in module.named_parameters() if param.requires_grad}]
    if frozen_missing:
        print(f"frozen base missing keys omitted from trainable checkpoint: {len(frozen_missing)}")
    noncritical_unexpected = [key for key in unexpected if key not in critical_unexpected]
    if noncritical_unexpected:
        print("non-critical unexpected keys while loading AnyFlow checkpoint:")
        for key in noncritical_unexpected:
            print(f"  {key}")
    return missing, unexpected
