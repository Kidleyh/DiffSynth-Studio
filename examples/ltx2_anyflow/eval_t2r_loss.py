import argparse
import csv
import json
import os
import sys

import torch
from safetensors.torch import load_file
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TRAINING_DIR = os.path.join(SCRIPT_DIR, "model_training")
if MODEL_TRAINING_DIR not in sys.path:
    sys.path.insert(0, MODEL_TRAINING_DIR)

from anyflow_ltx2 import flowmap_step, model_fn_ltx2_anyflow, scale_noise_at_timestep
from train import LTX2AnyFlowTrainingModule
from diffsynth.core import UnifiedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AnyFlow t->r reconstruction on cached LTX2 features.")
    parser.add_argument("--cache_dir", default="examples/ltx2_anyflow/eval/cache_val")
    parser.add_argument("--output_csv", default="examples/ltx2_anyflow/eval/t2r_loss.csv")
    parser.add_argument("--base_transformer_spec", default="DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["base=BASE"],
        help="Checkpoint specs like base=BASE trained=/path/to/step-1000.safetensors.",
    )
    parser.add_argument("--pairs", default="0.8:0.8,0.8:0.4,0.8:0.2,0.8:0.0,0.5:0.0")
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=float, default=25)
    return parser.parse_args()


def parse_checkpoint_specs(specs):
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid checkpoint spec: {spec}. Expected label=path or label=BASE.")
        label, path = spec.split("=", 1)
        parsed.append((label, path))
    return parsed


def parse_pairs(pairs):
    parsed = []
    for item in pairs.split(","):
        t_text, r_text = item.split(":", 1)
        t, r = float(t_text), float(r_text)
        if not (0.0 <= r <= t <= 1.0):
            raise ValueError(f"Invalid t:r pair {item}; expected 0 <= r <= t <= 1.")
        parsed.append((t, r))
    return parsed


def build_model(args, checkpoint_path):
    model = LTX2AnyFlowTrainingModule(
        model_id_with_origin_paths=args.base_transformer_spec,
        trainable_models="dit",
        extra_inputs="input_audio,input_image",
        use_gradient_checkpointing=False,
        task="flowmap_sft:train",
        device=args.device,
    )
    model.pipe.model_fn = model_fn_ltx2_anyflow
    model.to(args.device)
    model.eval()

    if checkpoint_path != "BASE":
        state = load_file(checkpoint_path, device="cpu")
        missing, unexpected = model.pipe.dit.load_state_dict(state, strict=False)
        print(
            f"Loaded {checkpoint_path}: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    return model


def prepare_cached_inputs(model, cached_inputs):
    inputs = model.transfer_data_to_device(cached_inputs, model.pipe.device, model.pipe.torch_dtype)
    for unit in model.pipe.units:
        inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    return inputs


def mse_or_none(pred, target):
    if pred is None or target is None:
        return None
    return torch.nn.functional.mse_loss(pred.float(), target.float()).detach()


def evaluate_pair(model, inputs_shared, t_frac, r_frac, seed):
    pipe = model.pipe
    device = pipe.device
    dtype = pipe.torch_dtype

    torch.manual_seed(seed)
    timestep = torch.tensor([pipe.scheduler.num_train_timesteps * t_frac], device=device, dtype=dtype)
    r_timestep = torch.tensor([pipe.scheduler.num_train_timesteps * r_frac], device=device, dtype=dtype)

    inputs = dict(inputs_shared)
    video_noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = scale_noise_at_timestep(pipe, inputs["input_latents"], video_noise, timestep)
    video_velocity_target = video_noise - inputs["input_latents"]
    target_video_r = scale_noise_at_timestep(pipe, inputs["input_latents"], video_noise, r_timestep)

    audio_velocity_target = None
    target_audio_r = None
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = scale_noise_at_timestep(pipe, inputs["audio_input_latents"], audio_noise, timestep)
        audio_velocity_target = audio_noise - inputs["audio_input_latents"]
        target_audio_r = scale_noise_at_timestep(pipe, inputs["audio_input_latents"], audio_noise, r_timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    with torch.no_grad():
        pred_video_velocity, pred_audio_velocity = pipe.model_fn(
            **models,
            **inputs,
            timestep=timestep,
            r_timestep=r_timestep,
        )
        pred_video_r = flowmap_step(pipe, inputs["video_latents"], pred_video_velocity, timestep, r_timestep)
        pred_audio_r = (
            flowmap_step(pipe, inputs["audio_latents"], pred_audio_velocity, timestep, r_timestep)
            if pred_audio_velocity is not None
            else None
        )

    return {
        "video_velocity_mse": mse_or_none(pred_video_velocity, video_velocity_target),
        "video_reconstruction_mse": mse_or_none(pred_video_r, target_video_r),
        "audio_velocity_mse": mse_or_none(pred_audio_velocity, audio_velocity_target),
        "audio_reconstruction_mse": mse_or_none(pred_audio_r, target_audio_r),
    }


def main():
    args = parse_args()
    checkpoints = parse_checkpoint_specs(args.checkpoints)
    pairs = parse_pairs(args.pairs)
    dataset = UnifiedDataset(base_path=args.cache_dir, metadata_path=None, repeat=1)
    if len(dataset) == 0:
        raise RuntimeError(f"No cached .pth files found in {args.cache_dir}. Run flowmap_sft:data_process first.")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    rows = []
    max_samples = min(args.max_samples, len(dataset))

    for label, checkpoint_path in checkpoints:
        model = build_model(args, checkpoint_path)
        for sample_id in tqdm(range(max_samples), desc=f"eval {label}"):
            cached_inputs = dataset[sample_id]
            inputs_shared, inputs_posi, inputs_nega = prepare_cached_inputs(model, cached_inputs)
            for pair_id, (t_frac, r_frac) in enumerate(pairs):
                metrics = evaluate_pair(
                    model,
                    inputs_shared,
                    t_frac,
                    r_frac,
                    seed=args.seed + sample_id * 1000 + pair_id,
                )
                row = {
                    "checkpoint": label,
                    "sample_id": sample_id,
                    "t": t_frac,
                    "r": r_frac,
                }
                row.update({k: None if v is None else float(v.cpu()) for k, v in metrics.items()})
                rows.append(row)
        del model
        torch.cuda.empty_cache()

    fieldnames = [
        "checkpoint",
        "sample_id",
        "t",
        "r",
        "video_velocity_mse",
        "video_reconstruction_mse",
        "audio_velocity_mse",
        "audio_reconstruction_mse",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for row in rows:
        key = (row["checkpoint"], row["t"], row["r"])
        bucket = summary.setdefault(key, {name: [] for name in fieldnames[4:]})
        for name in fieldnames[4:]:
            if row[name] is not None:
                bucket[name].append(row[name])
    summary_rows = []
    for (checkpoint, t_frac, r_frac), metrics in summary.items():
        item = {"checkpoint": checkpoint, "t": t_frac, "r": r_frac}
        for name, values in metrics.items():
            item[f"{name}_mean"] = sum(values) / len(values) if values else None
        summary_rows.append(item)
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
