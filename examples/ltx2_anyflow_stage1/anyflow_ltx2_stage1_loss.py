import torch
import torch.nn.functional as F


def _broadcast_time(t: torch.Tensor, x: torch.Tensor):
    if t.dtype == torch.bool:
        t = t.to(device=x.device)
    else:
        t = t.to(device=x.device, dtype=x.dtype)
    while t.ndim < x.ndim:
        t = t.view(*t.shape, 1)
    return t


def _per_sample_mean(x: torch.Tensor):
    if x.ndim == 1:
        return x
    return x.flatten(1).mean(dim=1)


def _prepare_loss_mask(mask, x: torch.Tensor):
    if mask is None:
        return None
    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(1)
    return mask.expand_as(x)


def _masked_per_sample_mean(x: torch.Tensor, mask):
    if mask is None:
        return _per_sample_mean(x)
    mask = _prepare_loss_mask(mask, x)
    numerator = (x * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def _mask_ratio(mask, x: torch.Tensor):
    if mask is None:
        return x.new_tensor(1.0)
    mask = _prepare_loss_mask(mask, x)
    return mask.float().mean()


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


def beta_time_weight(t: torch.Tensor):
    weight = torch.pow(t.clamp_min(1e-4), 1.0) * torch.pow((1.0 - t).clamp_min(1e-4), 0.5)
    return weight / weight.detach().mean().clamp_min(1e-6)


def _elementwise_regression(pred, target, loss_type="mse"):
    if loss_type == "huber":
        return F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    return F.mse_loss(pred.float(), target.float(), reduction="none")


def _zeros(batch_size, device):
    return torch.zeros(batch_size, device=device, dtype=torch.float32)


def _anyflow_guidance_fused(u_cond, u_uncond, guidance_scale):
    if guidance_scale <= 0:
        raise ValueError("cfg_scale/guidance scale must be > 0 for AnyFlow guidance-fused training.")
    return (u_cond - (1.0 - guidance_scale) * u_uncond.detach()) / guidance_scale


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
    use_time_weight=True,
    use_adaptive_weight=True,
    adaptive_eps=1e-6,
    cfg_fused=False,
    cfg_scale=1.0,
    negative_video_context=None,
    negative_audio_context=None,
    video_loss_mask=None,
    audio_loss_mask=None,
    **model_kwargs,
):
    device = video_latents.device
    batch_size = video_latents.shape[0]
    t, r, sampled_boundary_mask = sample_t_r(batch_size, device, boundary_prob, timestep_distribution)
    boundary_mask = (t - r).abs() < 1e-7

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
        rb_a = _broadcast_time(r, audio_latents)
        zt_a = (1.0 - tb_a) * audio_latents + tb_a * noise_a
        v_a = noise_a - audio_latents
    else:
        zt_a = None
        rb_a = None
        v_a = None

    def call_raw(v_latents, a_latents, t_value, r_value, v_ctx, a_ctx):
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

    def call(v_latents, a_latents, t_value, r_value):
        u_v_cond, u_a_cond = call_raw(v_latents, a_latents, t_value, r_value, video_context, audio_context)
        if not cfg_fused:
            return u_v_cond, u_a_cond
        if negative_video_context is None or negative_audio_context is None:
            raise ValueError("cfg_fused=True requires negative_video_context and negative_audio_context.")
        u_v_uncond, u_a_uncond = call_raw(
            v_latents,
            a_latents,
            t_value,
            r_value,
            negative_video_context,
            negative_audio_context,
        )
        u_v = _anyflow_guidance_fused(u_v_cond, u_v_uncond, cfg_scale)
        u_a = None
        if u_a_cond is not None and u_a_uncond is not None:
            u_a = _anyflow_guidance_fused(u_a_cond, u_a_uncond, cfg_scale)
        return u_v, u_a

    u_v, u_a = call(zt_v, zt_a, t, r)

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
        valid_v = _broadcast_time(fd_eps_t > 0, u_plus_v)
        du_dt_v = torch.where(valid_v, (u_plus_v - u_minus_v) / denom_v, torch.zeros_like(u_plus_v))
        u_tgt_v = v_v - (tb_v - rb_v) * du_dt_v
        if u_a is not None:
            denom_a = _broadcast_time((2.0 * fd_eps_t).clamp_min(1e-6), u_plus_a)
            valid_a = _broadcast_time(fd_eps_t > 0, u_plus_a)
            du_dt_a = torch.where(valid_a, (u_plus_a - u_minus_a) / denom_a, torch.zeros_like(u_plus_a))
            u_tgt_a = v_a - (_broadcast_time(t, audio_latents) - rb_a) * du_dt_a
        else:
            u_tgt_a = None

    video_loss_mask_prepared = _prepare_loss_mask(video_loss_mask, u_v) if video_loss_mask is not None else None
    reg_v = _masked_per_sample_mean(_elementwise_regression(u_v, u_tgt_v.detach(), loss_type), video_loss_mask_prepared)
    if u_a is not None and u_tgt_a is not None:
        audio_loss_mask_prepared = _prepare_loss_mask(audio_loss_mask, u_a) if audio_loss_mask is not None else None
        reg_a = _masked_per_sample_mean(_elementwise_regression(u_a, u_tgt_a.detach(), loss_type), audio_loss_mask_prepared)
    else:
        audio_loss_mask_prepared = None
        reg_a = _zeros(batch_size, device)
    reg_total = reg_v + float(audio_loss_weight) * reg_a

    time_weight = beta_time_weight(t) if use_time_weight else torch.ones_like(t)
    adaptive_fallback = not bool(boundary_mask.any())
    if use_adaptive_weight and not adaptive_fallback:
        mu_boundary = reg_total[boundary_mask].mean().detach()
        adaptive_weight = torch.ones_like(reg_total)
        non_boundary = ~boundary_mask
        adaptive_weight[non_boundary] = mu_boundary / (reg_total.detach()[non_boundary] + adaptive_eps)
    else:
        mu_boundary = reg_total.new_tensor(0.0)
        adaptive_weight = torch.ones_like(reg_total)

    loss_total = (time_weight * adaptive_weight * reg_total).mean()
    loss_video = (time_weight * adaptive_weight * reg_v).mean()
    loss_audio = (time_weight * adaptive_weight * reg_a).mean()

    non_boundary = ~boundary_mask
    boundary_reg_mean = reg_total[boundary_mask].mean() if boundary_mask.any() else reg_total.new_tensor(0.0)
    non_boundary_reg_mean = reg_total[non_boundary].mean() if non_boundary.any() else reg_total.new_tensor(0.0)
    logs = {
        "loss_total": loss_total.detach(),
        "loss_video": loss_video.detach(),
        "loss_audio": loss_audio.detach(),
        "t_mean": t.mean().detach(),
        "r_mean": r.mean().detach(),
        "delta_t_mean": (t - r).mean().detach(),
        "boundary_ratio": sampled_boundary_mask.float().mean().detach(),
        "u_norm_video": u_v.detach().float().norm(),
        "target_norm_video": u_tgt_v.detach().float().norm(),
        "u_norm_audio": u_a.detach().float().norm() if u_a is not None else loss_total.new_tensor(0.0),
        "target_norm_audio": u_tgt_a.detach().float().norm() if u_tgt_a is not None else loss_total.new_tensor(0.0),
        "time_weight_mean": time_weight.detach().mean(),
        "adaptive_weight_mean": adaptive_weight.detach().mean(),
        "mu_boundary": mu_boundary.detach(),
        "non_boundary_reg_mean": non_boundary_reg_mean.detach(),
        "boundary_reg_mean": boundary_reg_mean.detach(),
        "adaptive_fallback": loss_total.new_tensor(float(adaptive_fallback)),
        "video_loss_mask_ratio": _mask_ratio(video_loss_mask_prepared, u_v).detach(),
        "audio_loss_mask_ratio": _mask_ratio(audio_loss_mask_prepared, u_a).detach() if u_a is not None else loss_total.new_tensor(0.0),
        "using_video_loss_mask": loss_total.new_tensor(float(video_loss_mask_prepared is not None)),
        "using_audio_loss_mask": loss_total.new_tensor(float(audio_loss_mask_prepared is not None)),
    }
    return loss_total, logs
