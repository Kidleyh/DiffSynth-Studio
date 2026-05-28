from anyflow_ltx2_stage1_loss import anyflow_ltx2_stage1_loss


def _require(mapping, key, name):
    if key not in mapping or mapping[key] is None:
        available = sorted(mapping.keys())
        raise KeyError(f"Missing required {name} key '{key}'. Available {name} keys: {available}")
    return mapping[key]


def anyflow_stage1_native_loss(
    pipe,
    wrapper,
    inputs_shared,
    inputs_posi,
    audio_loss_weight=1.0,
    boundary_prob=0.5,
    fd_eps=1e-3,
    cfg_fused=False,
    cfg_scale=1.0,
    use_time_weight=True,
    use_adaptive_weight=True,
):
    """Adapt native LTX2 sft cache/unit outputs to AnyFlow Stage 1 loss."""

    video_latents = _require(inputs_shared, "input_latents", "inputs_shared")
    video_context = _require(inputs_posi, "video_context", "inputs_posi")
    audio_context = _require(inputs_posi, "audio_context", "inputs_posi")
    video_positions = _require(inputs_shared, "video_positions", "inputs_shared")
    audio_positions = _require(inputs_shared, "audio_positions", "inputs_shared")

    audio_latents = inputs_shared.get("audio_input_latents")

    negative_video_context = inputs_posi.get("negative_video_context")
    negative_audio_context = inputs_posi.get("negative_audio_context")
    if cfg_fused and (negative_video_context is None or negative_audio_context is None):
        available = sorted(inputs_posi.keys())
        raise KeyError(
            "cfg_fused=True requires negative_video_context and negative_audio_context. "
            f"Available inputs_posi keys: {available}"
        )

    return anyflow_ltx2_stage1_loss(
        wrapper,
        video_latents=video_latents,
        audio_latents=audio_latents,
        video_context=video_context,
        audio_context=audio_context,
        video_positions=video_positions,
        audio_positions=audio_positions,
        video_patchifier=inputs_shared.get("video_patchifier", getattr(pipe, "video_patchifier", None)),
        audio_patchifier=inputs_shared.get("audio_patchifier", getattr(pipe, "audio_patchifier", None)),
        audio_loss_weight=audio_loss_weight,
        boundary_prob=boundary_prob,
        fd_eps=fd_eps,
        cfg_fused=cfg_fused,
        cfg_scale=cfg_scale,
        negative_video_context=negative_video_context,
        negative_audio_context=negative_audio_context,
        use_time_weight=use_time_weight,
        use_adaptive_weight=use_adaptive_weight,
        use_gradient_checkpointing=inputs_shared.get("use_gradient_checkpointing", False),
        use_gradient_checkpointing_offload=inputs_shared.get("use_gradient_checkpointing_offload", False),
    )
