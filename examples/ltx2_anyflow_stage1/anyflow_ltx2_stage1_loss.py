import torch
import torch.nn.functional as F


def _broadcast_time(t: torch.Tensor, x: torch.Tensor):
    while t.ndim < x.ndim:
        t = t.view(*t.shape, 1)
    return t


def sample_t_r(batch_size, device, boundary_prob=0.5, distribution="uniform", min_delta=1e-4):
    boundary_mask = torch.rand(batch_size, device=device) < boundary_prob
    if distribution == "beta":
        beta = torch.distributions.Beta(
            torch.tensor(2.0, device=device),
            torch.tensor(1.5, device=device),
        )
        a = beta.sample((batch_size,))
        b = beta.sample((batch_size,))
    else:
        a = torch.rand(batch_size, device=device)
        b = torch.rand(batch_size, device=device)
    t = torch.maximum(a, b)
    r = torch.minimum(a, b)
    r = torch.where((t - r) < min_delta, (t - min_delta).clamp_min(0.0), r)
    r = torch.where(boundary_mask, t, r)
    return t, r, boundary_mask


def beta_loss_weight(t: torch.Tensor):
    weight = torch.pow(t.clamp_min(1e-4), 1.0) * torch.pow((1.0 - t).clamp_min(1e-4), 0.5)
    return weight / weight.detach().mean().clamp_min(1e-6)


def _loss(pred, target, loss_type="mse"):
    if pred is None or target is None:
        return pred.new_tensor(0.0) if pred is not None else target.new_tensor(0.0)
    if loss_type == "huber":
        return F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    return F.mse_loss(pred.float(), target.float(), reduction="none")


def _weighted_mean(loss, weight):
    while weight.ndim < loss.ndim:
        weight = weight.view(*weight.shape, 1)
    return (loss * weight).mean()


def anyflow_ltx2_stage1_loss(
    model,
    video_latents,
    audio_latents,
    video_context,
    audio_context,
    video_positions,
    audio_positions,
    video_patchifier=None,
    audio_patchifier=None,
    audio_loss_weight=1.0,
    boundary_prob=0.5,
    fd_eps=1e-3,
    loss_type="mse",
    timestep_distribution="uniform",
    adaptive_loss_weight=True,
    cfg_fused=False,
    cfg_scale=1.0,
    negative_video_context=None,
    negative_audio_context=None,
    **model_kwargs,
):
    device = video_latents.device
    batch_size = video_latents.shape[0]
    t, r, boundary_mask = sample_t_r(batch_size, device, boundary_prob, timestep_distribution)
    fd_eps_t = torch.full_like(t, fd_eps)
    fd_eps_t = torch.minimum(fd_eps_t, (1.0 - t).clamp_min(0.0))
    fd_eps_t = torch.minimum(fd_eps_t, (t - r).clamp_min(0.0))
    fd_eps_t = torch.where(fd_eps_t > 0, fd_eps_t.clamp_min(1e-6), torch.zeros_like(fd_eps_t))

    noise_v = torch.randn_like(video_latents)
    noise_a = torch.randn_like(audio_latents) if audio_latents is not None else None
    tb_v = _broadcast_time(t, video_latents)
    rb_v = _broadcast_time(r, video_latents)
    zt_v = (1.0 - tb_v) * video_latents + tb_v * noise_v
    v_v = noise_v - video_latents
    if audio_latents is not None:
        tb_a = _broadcast_time(t, audio_latents)
        zt_a = (1.0 - tb_a) * audio_latents + tb_a * noise_a
        v_a = noise_a - audio_latents
    else:
        zt_a = None
        v_a = None

    def call(v_latents, a_latents, t_value, r_value, v_ctx=video_context, a_ctx=audio_context):
        return model(
            video_latents=v_latents,
            audio_latents=a_latents,
            video_context=v_ctx,
            audio_context=a_ctx,
            video_positions=video_positions,
            audio_positions=audio_positions,
            timestep=t_value,
            r_timestep=r_value,
            video_patchifier=video_patchifier,
            audio_patchifier=audio_patchifier,
            **model_kwargs,
        )

    u_v, u_a = call(zt_v, zt_a, t, r)
    if cfg_fused and negative_video_context is not None:
        u_v_uncond, u_a_uncond = call(zt_v, zt_a, t, r, negative_video_context, negative_audio_context)
        u_v = u_v_uncond + cfg_scale * (u_v - u_v_uncond)
        if u_a is not None and u_a_uncond is not None:
            u_a = u_a_uncond + cfg_scale * (u_a - u_a_uncond)

    fd_v = _broadcast_time(fd_eps_t, zt_v)
    z_plus_v = zt_v + fd_v * v_v
    z_minus_v = zt_v - fd_v * v_v
    if zt_a is not None:
        fd_a = _broadcast_time(fd_eps_t, zt_a)
        z_plus_a = zt_a + fd_a * v_a
        z_minus_a = zt_a - fd_a * v_a
    else:
        z_plus_a = z_minus_a = None

    with torch.no_grad():
        u_plus_v, u_plus_a = call(z_plus_v, z_plus_a, (t + fd_eps_t).clamp_max(1.0), r)
        u_minus_v, u_minus_a = call(z_minus_v, z_minus_a, (t - fd_eps_t).clamp_min(0.0), r)
        denom_v = _broadcast_time((2.0 * fd_eps_t).clamp_min(1e-6), u_plus_v)
        du_dt_v = torch.where(_broadcast_time(fd_eps_t > 0, u_plus_v), (u_plus_v - u_minus_v) / denom_v, torch.zeros_like(u_plus_v))
        u_tgt_v = v_v - (tb_v - rb_v) * du_dt_v
        if u_a is not None:
            denom_a = _broadcast_time((2.0 * fd_eps_t).clamp_min(1e-6), u_plus_a)
            du_dt_a = torch.where(_broadcast_time(fd_eps_t > 0, u_plus_a), (u_plus_a - u_minus_a) / denom_a, torch.zeros_like(u_plus_a))
            rb_a = _broadcast_time(r, audio_latents)
            u_tgt_a = v_a - (_broadcast_time(t, audio_latents) - rb_a) * du_dt_a
        else:
            u_tgt_a = None

    weight = beta_loss_weight(t) if adaptive_loss_weight else torch.ones_like(t)
    loss_video = _weighted_mean(_loss(u_v, u_tgt_v.detach(), loss_type), weight)
    if u_a is not None and u_tgt_a is not None:
        loss_audio = _weighted_mean(_loss(u_a, u_tgt_a.detach(), loss_type), weight)
    else:
        loss_audio = loss_video.new_tensor(0.0)
    loss_total = loss_video + float(audio_loss_weight) * loss_audio

    logs = {
        "loss_total": loss_total.detach(),
        "loss_video": loss_video.detach(),
        "loss_audio": loss_audio.detach(),
        "t_mean": t.mean().detach(),
        "r_mean": r.mean().detach(),
        "delta_t_mean": (t - r).mean().detach(),
        "boundary_ratio": boundary_mask.float().mean().detach(),
        "u_norm_video": u_v.detach().float().norm(),
        "target_norm_video": u_tgt_v.detach().float().norm(),
        "u_norm_audio": u_a.detach().float().norm() if u_a is not None else loss_total.new_tensor(0.0),
        "target_norm_audio": u_tgt_a.detach().float().norm() if u_tgt_a is not None else loss_total.new_tensor(0.0),
    }
    return loss_total, logs

