import argparse
import csv
import os
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data import VideoData
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

from model_training.anyflow_ltx2 import attach_anyflow_time_adapters


NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, "
    "unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, "
    "extra limbs, disfigured hands, wrong hand count, inconsistent lighting direction, color banding, "
    "robotic voice, echo, background noise, off-sync audio, repetitive speech, jittery movement, AI artifacts."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fixed samples for base/trained LTX2 AnyFlow checkpoints.")
    parser.add_argument("--metadata_path", default="data/val.csv")
    parser.add_argument("--output_dir", default="examples/ltx2_anyflow/eval/generated_steps")
    parser.add_argument("--checkpoints", nargs="+", default=["base=BASE"])
    parser.add_argument("--steps", default="10,15,20,30,40")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--base_transformer_spec", default="DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors")
    return parser.parse_args()


def parse_checkpoint_specs(specs):
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid checkpoint spec: {spec}. Expected label=path or label=BASE.")
        label, path = spec.split("=", 1)
        parsed.append((label, path))
    return parsed


def read_rows(path, limit):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def transformer_model_config(spec, vram_config):
    if spec.endswith(".safetensors") and os.path.exists(spec):
        return ModelConfig(path=spec, **vram_config)
    model_id, pattern = spec.rsplit(":", 1)
    return ModelConfig(model_id=model_id, origin_file_pattern=pattern, **vram_config)


def build_pipe(args):
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="video_vae_decoder.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vae_decoder.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vocoder.safetensors", **vram_config),
            transformer_model_config(args.base_transformer_spec, vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
    )
    attach_anyflow_time_adapters(pipe.dit)
    return pipe


def load_checkpoint_overlay(pipe, checkpoint_path):
    if checkpoint_path == "BASE":
        return
    state = load_file(checkpoint_path, device="cpu")
    missing, unexpected = pipe.dit.load_state_dict(state, strict=False)
    print(f"Loaded {checkpoint_path}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def first_frame(video_path, height, width):
    frame = VideoData(video_path, height=height, width=width)[0]
    if isinstance(frame, Image.Image):
        return frame.convert("RGB").resize((width, height))
    return frame


def main():
    args = parse_args()
    checkpoints = parse_checkpoint_specs(args.checkpoints)
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    rows = read_rows(args.metadata_path, args.num_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as manifest_file:
        fieldnames = ["checkpoint", "sample_id", "steps", "seed", "prompt", "video", "input_audio", "output"]
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()

        for label, checkpoint_path in checkpoints:
            pipe = build_pipe(args)
            load_checkpoint_overlay(pipe, checkpoint_path)
            for sample_id, row in enumerate(tqdm(rows, desc=f"generate {label}")):
                image = first_frame(row["video"], args.height, args.width)
                prompt = row["prompt"]
                for step_count in steps:
                    sample_seed = args.seed + sample_id
                    output_path = output_dir / label / f"sample_{sample_id:03d}_steps_{step_count}.mp4"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    video, audio = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        seed=sample_seed,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        tiled=False,
                        input_images=[image],
                        input_images_indexes=[0],
                        input_images_strength=1.0,
                        num_inference_steps=step_count,
                    )
                    write_video_audio_ltx2(
                        video=video,
                        audio=audio,
                        output_path=str(output_path),
                        fps=args.fps,
                        audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
                    )
                    writer.writerow(
                        {
                            "checkpoint": label,
                            "sample_id": sample_id,
                            "steps": step_count,
                            "seed": sample_seed,
                            "prompt": prompt,
                            "video": row.get("video"),
                            "input_audio": row.get("input_audio"),
                            "output": str(output_path),
                        }
                    )
                    manifest_file.flush()
            del pipe
            torch.cuda.empty_cache()
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
