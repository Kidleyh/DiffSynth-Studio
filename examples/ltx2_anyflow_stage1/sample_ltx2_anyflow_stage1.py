import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from anyflow_ltx2_lora import inject_lora_linear, parse_name_filter
from anyflow_ltx2_model_wrapper import LTX2AnyFlowWrapper, load_trainable_state_dict
from anyflow_ltx2_scheduler import FlowMapEulerSchedulerForLTX2


def dtype_from_arg(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_model_configs(path):
    if path is None:
        return []
    from diffsynth.pipelines.ltx2_audio_video import ModelConfig

    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("model_configs", [data])
    return [ModelConfig(**item) for item in data]


def checkpoint_paths(checkpoint):
    ckpt = Path(checkpoint)
    if ckpt.is_file():
        return ckpt.parent, ckpt, None
    return ckpt, ckpt / "anyflow_wrapper.pt", ckpt / "anyflow_config.json"


def load_anyflow_config(checkpoint):
    ckpt_dir, _, cfg_path = checkpoint_paths(checkpoint)
    if cfg_path is None or not cfg_path.exists():
        raise RuntimeError(f"Missing anyflow_config.json in checkpoint directory: {ckpt_dir}")
    return json.loads(cfg_path.read_text())


def load_anyflow_state(checkpoint):
    _, state_path, _ = checkpoint_paths(checkpoint)
    return torch.load(state_path, map_location="cpu")


def state_dict_has_lora(state):
    return any("lora_A" in key or "lora_B" in key for key in state.keys())


def resolve_lora_settings(cfg, state=None):
    state_has_lora = state_dict_has_lora(state) if state is not None else False
    lora_base_model = cfg.get("lora_base_model", None)
    if lora_base_model not in (None, "", "dit"):
        raise RuntimeError(
            "AnyFlow Stage 1 sampling currently supports LoRA injection on DiT only, "
            f"but checkpoint config has lora_base_model={lora_base_model!r}."
        )

    enabled = bool(cfg.get("use_lora", False)) or lora_base_model == "dit" or state_has_lora
    rank = int(cfg.get("lora_rank", 256))
    alpha = float(cfg.get("lora_alpha", cfg.get("lora_scale", rank)))
    if cfg.get("lora_target_modules") is not None:
        target_modules = parse_name_filter(cfg.get("lora_target_modules"))
        source = "native-config"
    elif cfg.get("lora_target_filter") is not None:
        target_modules = parse_name_filter(cfg.get("lora_target_filter"))
        source = "legacy-config"
    else:
        target_modules = parse_name_filter("to_k,to_q,to_v,to_out.0")
        if lora_base_model == "dit":
            source = "native-config"
        elif bool(cfg.get("use_lora", False)):
            source = "legacy-config"
        elif state_has_lora:
            source = "state-dict"
        else:
            source = "disabled"
    if not enabled:
        source = "disabled"
    return {
        "enabled": enabled,
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules,
        "source": source,
    }


def build_wrapper(pipe, cfg, device, dtype, state=None):
    wrapper_cfg = cfg.get("wrapper_config", {})
    wrapper = LTX2AnyFlowWrapper(
        pipe.dit,
        gate=float(wrapper_cfg.get("gate", cfg.get("gate_init", 0.25))),
        freeze_base=bool(wrapper_cfg.get("freeze_base", True)),
    )
    lora = resolve_lora_settings(cfg, state=state)
    if lora["enabled"]:
        updated = inject_lora_linear(
            wrapper.dit,
            rank=lora["rank"],
            alpha=lora["alpha"],
            name_filter=lora["target_modules"],
        )
        if updated == 0:
            raise RuntimeError(
                "Checkpoint/config/state requested LoRA, but no target Linear layers matched "
                f"targets={lora['target_modules']} rank={lora['rank']} alpha={lora['alpha']}."
            )
        print(
            f"Injected LoRA into {updated} linear layers from {lora['source']}: "
            f"rank={lora['rank']}, alpha={lora['alpha']}, targets={lora['target_modules']}",
            flush=True,
        )
    return wrapper.to(device=device, dtype=dtype)


def load_anyflow_checkpoint(wrapper, checkpoint=None, state=None):
    if state is None:
        if checkpoint is None:
            raise ValueError("Either checkpoint or state must be provided to load_anyflow_checkpoint.")
        state = load_anyflow_state(checkpoint)
    load_trainable_state_dict(wrapper, state, strict_trainable=True)


def prepare_pipeline_inputs(pipe, args, device):
    pipe.scheduler.set_timesteps(1)
    inputs_posi = {"prompt": args.prompt}
    inputs_nega = {"negative_prompt": args.negative_prompt}
    inputs_shared = {
        "input_images": None,
        "input_images_indexes": [0],
        "input_images_strength": 1.0,
        "retake_video": None,
        "retake_video_regions": None,
        "retake_audio": None,
        "retake_audio_regions": None,
        "in_context_videos": None,
        "in_context_downsample_factor": 2,
        "seed": args.seed,
        "rand_device": device,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "cfg_scale": args.cfg_scale,
        "tiled": args.tiled,
        "tile_size_in_pixels": args.tile_size_in_pixels,
        "tile_overlap_in_pixels": args.tile_overlap_in_pixels,
        "tile_size_in_frames": args.tile_size_in_frames,
        "tile_overlap_in_frames": args.tile_overlap_in_frames,
        "use_two_stage_pipeline": False,
        "use_distilled_pipeline": False,
        "clear_lora_before_state_two": False,
        "stage2_spatial_upsample_factor": 2,
        "video_patchifier": pipe.video_patchifier,
        "audio_patchifier": pipe.audio_patchifier,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
    return inputs_shared, inputs_posi


def load_init_latents(init_latents, device, dtype):
    if init_latents is None:
        return None, None
    base = Path(init_latents)
    if base.is_dir():
        video_path = base / "video_latents.pt"
        audio_path = base / "audio_latents.pt"
    else:
        data = json.loads(base.read_text())
        video_path = Path(data["video_latents"])
        audio_path = Path(data["audio_latents"])
    video_latents = torch.load(video_path, map_location=device).to(dtype=dtype)
    audio_latents = torch.load(audio_path, map_location=device).to(dtype=dtype)
    return video_latents, audio_latents


def norm_value(x):
    return float(x.detach().float().norm().cpu())


def save_latent_rollout(output_path, video_latents, audio_latents, stats):
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(video_latents.detach().cpu(), out / "final_video_latents.pt")
    torch.save(audio_latents.detach().cpu(), out / "final_audio_latents.pt")
    (out / "rollout_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"latent rollout saved to {out}", flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_path", default="anyflow_stage1_sample.mp4")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=float, default=24)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--num_inference_steps", type=int, choices=[4, 8, 16, 32], default=8)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--tiled", action="store_true", default=True)
    parser.add_argument("--tile_size_in_pixels", type=int, default=512)
    parser.add_argument("--tile_overlap_in_pixels", type=int, default=128)
    parser.add_argument("--tile_size_in_frames", type=int, default=128)
    parser.add_argument("--tile_overlap_in_frames", type=int, default=24)
    parser.add_argument("--latent_rollout_only", action="store_true")
    parser.add_argument("--init_latents")
    args = parser.parse_args()

    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline
    from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = dtype_from_arg(args.dtype) if device == "cuda" else torch.float32
    cfg = load_anyflow_config(args.checkpoint)
    state = load_anyflow_state(args.checkpoint)
    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=load_model_configs(args.model_config_path),
    )
    wrapper = build_wrapper(pipe, cfg, device, dtype, state=state)
    load_anyflow_checkpoint(wrapper, state=state)
    wrapper.eval()

    inputs_shared, inputs_posi = prepare_pipeline_inputs(pipe, args, device)
    init_video, init_audio = load_init_latents(args.init_latents, device, dtype)
    video_latents = init_video if init_video is not None else inputs_shared["video_latents"]
    audio_latents = init_audio if init_audio is not None else inputs_shared["audio_latents"]
    scheduler = FlowMapEulerSchedulerForLTX2(device=device, dtype=dtype)
    timesteps = scheduler.set_timesteps(args.num_inference_steps, device=device, dtype=dtype)
    video_context = inputs_posi["video_context"]
    audio_context = inputs_posi["audio_context"]

    stats = {
        "num_inference_steps": args.num_inference_steps,
        "video_latent_shape": list(video_latents.shape),
        "audio_latent_shape": list(audio_latents.shape),
        "initial_video_norm": norm_value(video_latents),
        "initial_audio_norm": norm_value(audio_latents),
        "steps": [],
    }
    for i in tqdm(range(args.num_inference_steps)):
        t = timesteps[i].expand(video_latents.shape[0])
        r = timesteps[i + 1].expand(video_latents.shape[0])
        prev_video = video_latents
        prev_audio = audio_latents
        u_video, u_audio = wrapper(
            video_latents=video_latents,
            audio_latents=audio_latents,
            video_context=video_context,
            audio_context=audio_context,
            video_positions=inputs_shared["video_positions"],
            audio_positions=inputs_shared["audio_positions"],
            timestep=t,
            r_timestep=r,
            video_patchifier=pipe.video_patchifier,
            audio_patchifier=pipe.audio_patchifier,
        )
        video_latents = scheduler.step(u_video, video_latents, t, r)
        audio_latents = scheduler.step(u_audio, audio_latents, t, r)
        stats["steps"].append(
            {
                "index": i,
                "t": float(timesteps[i].detach().cpu()),
                "r": float(timesteps[i + 1].detach().cpu()),
                "video_norm": norm_value(video_latents),
                "audio_norm": norm_value(audio_latents),
                "video_delta_norm": norm_value(video_latents - prev_video),
                "audio_delta_norm": norm_value(audio_latents - prev_audio),
            }
        )
    stats["final_video_norm"] = norm_value(video_latents)
    stats["final_audio_norm"] = norm_value(audio_latents)
    stats["schedule"] = [float(x) for x in timesteps.detach().cpu()]

    if args.latent_rollout_only:
        save_latent_rollout(args.output_path, video_latents, audio_latents, stats)
        return

    video = pipe.video_vae_decoder.decode(
        video_latents,
        args.tiled,
        args.tile_size_in_pixels,
        args.tile_overlap_in_pixels,
        args.tile_size_in_frames,
        args.tile_overlap_in_frames,
    )
    video = pipe.vae_output_to_video(video)
    decoded_audio = pipe.audio_vae_decoder(audio_latents)
    decoded_audio = pipe.audio_vocoder(decoded_audio)
    decoded_audio = pipe.output_audio_format_check(decoded_audio)
    write_video_audio_ltx2(video=video, audio=decoded_audio, output_path=args.output_path, fps=args.frame_rate, audio_sample_rate=24000)


if __name__ == "__main__":
    main()

