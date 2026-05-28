from anyflow_ltx2_stage1_loss import anyflow_ltx2_stage1_loss


OPTIONAL_CONDITION_KEYS = (
    "input_latents_video",
    "denoise_mask_video",
    "ref_frames_latents",
    "ref_frames_positions",
    "in_context_video_latents",
    "in_context_video_positions",
    "input_latents_audio",
    "denoise_mask_audio",
)

VIDEO_CONDITION_KEYS = (
    "input_latents_video",
    "denoise_mask_video",
    "ref_frames_latents",
    "ref_frames_positions",
    "in_context_video_latents",
    "in_context_video_positions",
)

AUDIO_CONDITION_KEYS = (
    "input_latents_audio",
    "denoise_mask_audio",
)

REF_CONDITION_KEYS = (
    "ref_frames_latents",
    "ref_frames_positions",
    "in_context_video_latents",
    "in_context_video_positions",
)

REQUIRED_KEY_SPECS = (
    ("inputs_shared", "input_latents"),
    ("inputs_shared", "video_positions"),
    ("inputs_shared", "audio_positions"),
    ("inputs_posi", "video_context"),
    ("inputs_posi", "audio_context"),
)


def _available(inputs_shared, inputs_posi, inputs_nega):
    return (
        f"Available inputs_shared keys: {sorted(inputs_shared.keys())}; "
        f"inputs_posi keys: {sorted(inputs_posi.keys())}; "
        f"inputs_nega keys: {sorted(inputs_nega.keys()) if inputs_nega is not None else []}"
    )


def _require(mapping, key, name, inputs_shared, inputs_posi, inputs_nega):
    if mapping is None or key not in mapping or mapping[key] is None:
        raise KeyError(
            f"Missing required AnyFlow native key '{key}' in {name}. "
            + _available(inputs_shared, inputs_posi, inputs_nega)
        )
    return mapping[key]


def _present(mapping, key):
    return mapping is not None and key in mapping and mapping[key] is not None


def _mapping_by_name(name, inputs_shared, inputs_posi, inputs_nega):
    if name == "inputs_shared":
        return inputs_shared
    if name == "inputs_posi":
        return inputs_posi
    if name == "inputs_nega":
        return inputs_nega
    raise KeyError(name)


def _qualified(name, key):
    return f"{name}.{key}"


def _make_native_key_report(
    inputs_shared,
    inputs_posi,
    inputs_nega,
    cfg_fused=False,
    audio_present=False,
    audio_fallback_reason=None,
):
    required_found = []
    required_missing = []
    for mapping_name, key in REQUIRED_KEY_SPECS:
        mapping = _mapping_by_name(mapping_name, inputs_shared, inputs_posi, inputs_nega)
        target = required_found if _present(mapping, key) else required_missing
        target.append(_qualified(mapping_name, key))

    optional_found = [key for key in OPTIONAL_CONDITION_KEYS if _present(inputs_shared, key)]
    optional_missing = [key for key in OPTIONAL_CONDITION_KEYS if key not in optional_found]
    video_used = [key for key in VIDEO_CONDITION_KEYS if key in optional_found]
    audio_used = [key for key in AUDIO_CONDITION_KEYS if key in optional_found]
    ref_used = [key for key in REF_CONDITION_KEYS if key in optional_found]

    negative_video_context_present = _present(inputs_nega, "video_context")
    negative_audio_context_present = _present(inputs_nega, "audio_context")
    using_negative_context = bool(cfg_fused and negative_video_context_present and negative_audio_context_present)
    audio_condition_present = bool(audio_used)

    return {
        "required_keys_found": required_found,
        "required_keys_missing": required_missing,
        "optional_condition_keys_found": optional_found,
        "optional_condition_keys_missing": optional_missing,
        "video_target_key": "inputs_shared.input_latents" if _present(inputs_shared, "input_latents") else None,
        "audio_target_key": "inputs_shared.audio_input_latents" if audio_present else None,
        "video_context_key": "inputs_posi.video_context" if _present(inputs_posi, "video_context") else None,
        "audio_context_key": "inputs_posi.audio_context" if _present(inputs_posi, "audio_context") else None,
        "negative_video_context_key": "inputs_nega.video_context" if using_negative_context else None,
        "negative_audio_context_key": "inputs_nega.audio_context" if using_negative_context else None,
        "video_condition_keys_used_in_forward": video_used,
        "audio_condition_keys_used_in_forward": audio_used,
        "ref_condition_keys_used_in_forward": ref_used,
        "using_video_conditioning": bool(video_used),
        "using_audio_conditioning": bool(audio_used),
        "using_ref_frame_conditioning": bool(ref_used),
        "negative_context_available": bool(negative_video_context_present or negative_audio_context_present),
        "using_negative_context": using_negative_context,
        "audio_present": bool(audio_present),
        "audio_target_present": bool(audio_present),
        "audio_condition_present": audio_condition_present,
        "audio_fallback_reason": audio_fallback_reason,
        "available_inputs_shared_keys": sorted(inputs_shared.keys()),
        "available_inputs_posi_keys": sorted(inputs_posi.keys()),
        "available_inputs_nega_keys": sorted(inputs_nega.keys()) if inputs_nega is not None else [],
    }


def anyflow_stage1_native_loss(
    pipe,
    wrapper,
    inputs_shared,
    inputs_posi,
    inputs_nega=None,
    audio_loss_weight=1.0,
    boundary_prob=0.5,
    fd_eps=1e-3,
    cfg_fused=False,
    cfg_scale=1.0,
    use_time_weight=True,
    use_adaptive_weight=True,
):
    """Adapt native LTX2 sft cache/unit outputs to AnyFlow Stage 1 loss."""

    inputs_nega = inputs_nega or {}

    video_latents = _require(inputs_shared, "input_latents", "inputs_shared", inputs_shared, inputs_posi, inputs_nega)
    video_context = _require(inputs_posi, "video_context", "inputs_posi", inputs_shared, inputs_posi, inputs_nega)
    audio_context = _require(inputs_posi, "audio_context", "inputs_posi", inputs_shared, inputs_posi, inputs_nega)
    video_positions = _require(inputs_shared, "video_positions", "inputs_shared", inputs_shared, inputs_posi, inputs_nega)
    audio_positions = _require(inputs_shared, "audio_positions", "inputs_shared", inputs_shared, inputs_posi, inputs_nega)

    audio_latents = inputs_shared.get("audio_input_latents")
    audio_present = audio_latents is not None
    audio_condition_present = any(_present(inputs_shared, key) for key in AUDIO_CONDITION_KEYS)
    if audio_present:
        audio_fallback_reason = None
    elif audio_condition_present:
        audio_fallback_reason = (
            "audio condition keys were present but inputs_shared.audio_input_latents is missing, "
            "so audio target flow loss is disabled"
        )
    else:
        audio_fallback_reason = "audio_input_latents missing from native cache/unit outputs"

    native_key_report = _make_native_key_report(
        inputs_shared,
        inputs_posi,
        inputs_nega,
        cfg_fused=cfg_fused,
        audio_present=audio_present,
        audio_fallback_reason=audio_fallback_reason,
    )
    if native_key_report["required_keys_missing"]:
        raise KeyError(
            "Missing required AnyFlow native keys: "
            f"{native_key_report['required_keys_missing']}. "
            + _available(inputs_shared, inputs_posi, inputs_nega)
        )

    negative_video_context = None
    negative_audio_context = None
    if cfg_fused:
        negative_video_context = _require(inputs_nega, "video_context", "inputs_nega", inputs_shared, inputs_posi, inputs_nega)
        negative_audio_context = _require(inputs_nega, "audio_context", "inputs_nega", inputs_shared, inputs_posi, inputs_nega)

    condition_kwargs = {key: inputs_shared.get(key) for key in OPTIONAL_CONDITION_KEYS}

    loss, logs = anyflow_ltx2_stage1_loss(
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
        video_loss_mask=inputs_shared.get("denoise_mask_video"),
        audio_loss_mask=inputs_shared.get("denoise_mask_audio"),
        use_gradient_checkpointing=inputs_shared.get("use_gradient_checkpointing", False),
        use_gradient_checkpointing_offload=inputs_shared.get("use_gradient_checkpointing_offload", False),
        **condition_kwargs,
    )

    logs.update(
        {
            "audio_present": audio_present,
            "audio_target_present": audio_present,
            "audio_condition_present": audio_condition_present,
            "audio_fallback_reason": audio_fallback_reason,
            "audio_loss_weight": float(audio_loss_weight),
            "using_video_conditioning": native_key_report["using_video_conditioning"],
            "using_audio_conditioning": native_key_report["using_audio_conditioning"],
            "using_ref_frame_conditioning": native_key_report["using_ref_frame_conditioning"],
            "using_negative_context": native_key_report["using_negative_context"],
        }
    )
    if not audio_present:
        logs["loss_audio"] = logs["loss_audio"] * 0.0
        logs["u_norm_audio"] = logs["u_norm_audio"] * 0.0
        logs["target_norm_audio"] = logs["target_norm_audio"] * 0.0
        logs["audio_loss_mask_ratio"] = logs["audio_loss_mask_ratio"] * 0.0
        logs["using_audio_loss_mask"] = logs["using_audio_loss_mask"] * 0.0

    return loss, logs, native_key_report
