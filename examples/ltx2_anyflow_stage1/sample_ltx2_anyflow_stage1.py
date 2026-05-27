import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from anyflow_ltx2_model_wrapper import LTX2AnyFlowWrapper
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


def load_anyflow_checkpoint(wrapper, checkpoint):
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "anyflow_wrapper.pt"
    state = torch.load(path, map_location="cpu")
    missing, unexpected = wrapper.load_state_dict(state, strict=False)
    print(f"loaded AnyFlow checkpoint: missing={len(missing)} unexpected={len(unexpected)}", flush=True)


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
    return inputs_shared, inputs_posi, inputs_nega


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
    args = parser.parse_args()

    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline
    from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = dtype_from_arg(args.dtype) if device == "cuda" else torch.float32
    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=load_model_configs(args.model_config_path),
    )
    wrapper = LTX2AnyFlowWrapper(pipe.dit, freeze_base=True).to(device=device, dtype=dtype)
    load_anyflow_checkpoint(wrapper, args.checkpoint)
    wrapper.eval()

    inputs_shared, inputs_posi, _ = prepare_pipeline_inputs(pipe, args, device)
    video_latents = inputs_shared["video_latents"]
    audio_latents = inputs_shared["audio_latents"]
    scheduler = FlowMapEulerSchedulerForLTX2(device=device, dtype=dtype)
    timesteps = scheduler.set_timesteps(args.num_inference_steps, device=device, dtype=dtype)
    video_context = inputs_posi["video_context"]
    audio_context = inputs_posi["audio_context"]

    for i in tqdm(range(args.num_inference_steps)):
        t = timesteps[i].expand(video_latents.shape[0])
        r = timesteps[i + 1].expand(video_latents.shape[0])
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

