import argparse
import csv
import json
import os
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from anyflow_ltx2_lora import (
    inject_lora_linear,
    looks_like_tiny_time_only_trainable,
    parse_name_filter,
    trainable_parameter_report,
)
from anyflow_ltx2_debug import collect_gradient_sanity
from anyflow_ltx2_model_wrapper import LTX2AnyFlowWrapper, load_trainable_state_dict, trainable_state_dict
from anyflow_ltx2_stage1_loss import anyflow_ltx2_stage1_loss


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


def checkpoint_dir(output_dir, step):
    return Path(output_dir) / f"checkpoint-step_{step:06d}"


def anyflow_config_from_args(args):
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank
    return {
        "use_lora": bool(args.use_lora),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(lora_alpha),
        "lora_target_filter": list(parse_name_filter(args.lora_target_filter)),
        "gate_init": float(args.gate_init),
        "audio_loss_weight": float(args.audio_loss_weight),
        "boundary_prob": float(args.boundary_prob),
        "fd_eps": float(args.fd_eps),
        "cfg_fused": bool(args.cfg_fused),
        "cfg_scale": float(args.cfg_scale),
        "use_time_weight": not bool(args.disable_time_weight),
        "use_adaptive_weight": not bool(args.disable_adaptive_weight),
        "wrapper_class": "LTX2AnyFlowWrapper",
        "wrapper_config": {"gate": float(args.gate_init), "freeze_base": True},
        "gradient_sanity_checked": bool(getattr(args, "_gradient_sanity_checked", False)),
        "frozen_unused_r_adaln_linear": True,
        "trainable_without_grad_policy": "allow" if args.allow_trainable_without_grad else "raise",
    }


class LatentCacheDataset(Dataset):
    def __init__(self, metadata, latent_cache_dir=None):
        self.latent_cache_dir = Path(latent_cache_dir) if latent_cache_dir else None
        path = Path(metadata)
        if path.suffix == ".jsonl":
            self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            with path.open() as f:
                self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def _resolve(self, value):
        p = Path(value)
        if not p.is_absolute() and self.latent_cache_dir is not None:
            p = self.latent_cache_dir / p
        return p

    def __getitem__(self, idx):
        row = dict(self.rows[idx])
        item = {
            "video_latents": torch.load(self._resolve(row["video_latents"]), map_location="cpu"),
            "audio_latents": torch.load(self._resolve(row["audio_latents"]), map_location="cpu"),
            "video_positions": torch.load(self._resolve(row["video_positions"]), map_location="cpu"),
            "audio_positions": torch.load(self._resolve(row["audio_positions"]), map_location="cpu"),
            "prompt": row.get("prompt", ""),
        }
        if row.get("prompt_embeds"):
            embeds = torch.load(self._resolve(row["prompt_embeds"]), map_location="cpu")
            item["video_context"] = embeds
            item["audio_context"] = embeds
        if row.get("video_context"):
            item["video_context"] = torch.load(self._resolve(row["video_context"]), map_location="cpu")
        if row.get("audio_context"):
            item["audio_context"] = torch.load(self._resolve(row["audio_context"]), map_location="cpu")
        if row.get("negative_video_context"):
            item["negative_video_context"] = torch.load(self._resolve(row["negative_video_context"]), map_location="cpu")
        if row.get("negative_audio_context"):
            item["negative_audio_context"] = torch.load(self._resolve(row["negative_audio_context"]), map_location="cpu")
        return item


def collate(batch):
    out = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        out[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return out


class TinyAnyFlowModel(torch.nn.Module):
    def __init__(self, channels_v=4, channels_a=4):
        super().__init__()
        self.video = torch.nn.Conv3d(channels_v, channels_v, 1)
        self.audio = torch.nn.Conv2d(channels_a, channels_a, 1)
        self.time = torch.nn.Linear(2, channels_v + channels_a)

    def forward(self, video_latents, audio_latents, timestep, r_timestep, **kwargs):
        b = video_latents.shape[0]
        t = torch.as_tensor(timestep, device=video_latents.device).flatten()
        r = torch.as_tensor(r_timestep, device=video_latents.device).flatten()
        if t.numel() == 1:
            t = t.expand(b)
        if r.numel() == 1:
            r = r.expand(b)
        tr = torch.stack([t[:b], r[:b]], dim=-1).float()
        bias = self.time(tr)
        bv = bias[:, : video_latents.shape[1]].view(b, video_latents.shape[1], 1, 1, 1).to(video_latents.dtype)
        ba = bias[:, -audio_latents.shape[1]:].view(b, audio_latents.shape[1], 1, 1).to(audio_latents.dtype)
        return self.video(video_latents) + bv, self.audio(audio_latents) + ba


def build_real_model(args, device, dtype):
    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline

    if args.model_config_path is None:
        raise RuntimeError("--model_config_path is required for real LTX2 wrapper training or --smoke_test_real_wrapper.")
    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=load_model_configs(args.model_config_path),
    )
    wrapper = LTX2AnyFlowWrapper(pipe.dit, gate=args.gate_init, freeze_base=True)
    if args.use_lora:
        updated = inject_lora_linear(
            wrapper.dit,
            rank=args.lora_rank,
            alpha=args.lora_alpha if args.lora_alpha is not None else args.lora_rank,
            name_filter=parse_name_filter(args.lora_target_filter),
        )
        print(f"injected LoRA into {updated} linear layers", flush=True)
        if updated == 0:
            raise RuntimeError("use_lora=True but no Linear layers matched --lora_target_filter.")
    return pipe, wrapper


def print_trainable_report(model, allow_tiny=False, tiny_ok=False):
    report = trainable_parameter_report(model)
    print(f"total params: {report['total']:,}", flush=True)
    print(f"trainable params: {report['trainable']:,}", flush=True)
    print(f"trainable ratio: {report['ratio']:.8f}", flush=True)
    print("trainable module name sample:", flush=True)
    for name in report["name_sample"]:
        print(f"  {name}", flush=True)
    if not tiny_ok and not allow_tiny:
        if report["trainable"] < 1_000_000 or looks_like_tiny_time_only_trainable(report["names"]):
            raise RuntimeError(
                "Trainable parameter set is too small for Stage 1 flow-map training. "
                "Use --use_lora --lora_rank 256, or pass --allow_tiny_trainable_params for debugging only."
            )
    return report


def get_prompt_unit(pipe):
    return next(unit for unit in pipe.units if unit.__class__.__name__ == "LTX2AudioVideoUnit_PromptEmbedder")


def encode_prompts(pipe, prompts):
    prompt_unit = get_prompt_unit(pipe)
    video_context, audio_context = [], []
    with torch.no_grad():
        for prompt in prompts:
            vc, ac, _ = prompt_unit.encode_prompt(pipe, prompt)
            video_context.append(vc.squeeze(0).detach().cpu())
            audio_context.append(ac.squeeze(0).detach().cpu())
    return torch.stack(video_context), torch.stack(audio_context)


def ensure_contexts(pipe, batch, cfg_fused):
    if "video_context" not in batch or "audio_context" not in batch:
        prompts = batch.get("prompt")
        if prompts is None:
            raise RuntimeError("Missing cached context tensors and no prompt column was provided.")
        vc, ac = encode_prompts(pipe, prompts)
        batch["video_context"] = vc
        batch["audio_context"] = ac
    if cfg_fused and ("negative_video_context" not in batch or "negative_audio_context" not in batch):
        batch_size = len(batch["prompt"]) if "prompt" in batch else int(batch["video_context"].shape[0])
        vc, ac = encode_prompts(pipe, [""] * batch_size)
        batch["negative_video_context"] = vc
        batch["negative_audio_context"] = ac
    return batch


def prepare_pipeline_inputs(pipe, args, device):
    pipe.scheduler.set_timesteps(1)
    inputs_posi = {"prompt": args.prompt}
    inputs_nega = {"negative_prompt": ""}
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
        "seed": 43,
        "rand_device": device,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "cfg_scale": 1.0,
        "tiled": False,
        "tile_size_in_pixels": 512,
        "tile_overlap_in_pixels": 128,
        "tile_size_in_frames": 128,
        "tile_overlap_in_frames": 24,
        "use_two_stage_pipeline": False,
        "use_distilled_pipeline": False,
        "clear_lora_before_state_two": False,
        "stage2_spatial_upsample_factor": 2,
        "video_patchifier": pipe.video_patchifier,
        "audio_patchifier": pipe.audio_patchifier,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
    if args.cfg_fused:
        empty_v, empty_a = encode_prompts(pipe, [""])
        inputs_posi["negative_video_context"] = empty_v
        inputs_posi["negative_audio_context"] = empty_a
    return inputs_shared, inputs_posi


def move_batch(batch, device, dtype):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
        else:
            moved[key] = value
    return moved


def save_checkpoint(output_dir, step, model, optimizer, args):
    ckpt = checkpoint_dir(output_dir, step)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(trainable_state_dict(model), ckpt / "anyflow_wrapper.pt")
    torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
    (ckpt / "training_state.json").write_text(json.dumps({"global_step": step, "args": vars(args)}, indent=2))
    (ckpt / "anyflow_config.json").write_text(json.dumps(anyflow_config_from_args(args), indent=2))
    return ckpt


def load_checkpoint(model, optimizer, resume_dir):
    resume_dir = Path(resume_dir)
    state = torch.load(resume_dir / "anyflow_wrapper.pt", map_location="cpu")
    missing, unexpected = load_trainable_state_dict(model, state, strict_trainable=True)
    optimizer.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location="cpu"))
    meta = json.loads((resume_dir / "training_state.json").read_text())
    print(
        "checkpoint load summary: "
        f"missing={len(missing)} unexpected={len(unexpected)} global_step={meta.get('global_step', meta.get('step', 0))}",
        flush=True,
    )
    return int(meta.get("global_step", meta.get("step", 0)))


def save_gradient_sanity_report(output_dir, step, report):
    path = Path(output_dir) / f"gradient_sanity_step_{step:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path


def run_gradient_sanity_or_raise(model, args, step):
    report = collect_gradient_sanity(model)
    print(f"gradient_sanity_step_{step:06d}: {json.dumps(report, indent=2)}", flush=True)
    if args.save_gradient_sanity:
        save_gradient_sanity_report(args.output_dir, step, report)
    if report["trainable_zero_grad_names"]:
        warnings.warn(
            "Some trainable tensors have zero gradients: "
            + ", ".join(report["trainable_zero_grad_names"][:20])
        )
    if report["trainable_without_grad_names"] and not args.allow_trainable_without_grad:
        raise RuntimeError(
            "Trainable tensors without gradients were found. "
            "Pass --allow_trainable_without_grad to continue for debugging. "
            f"First names: {report['trainable_without_grad_names'][:20]}"
        )
    args._gradient_sanity_checked = True
    return report


def apply_resume_config(args):
    if not args.resume:
        return
    cfg_path = Path(args.resume) / "anyflow_config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    args.use_lora = bool(cfg.get("use_lora", args.use_lora))
    args.lora_rank = int(cfg.get("lora_rank", args.lora_rank))
    args.lora_alpha = float(cfg.get("lora_alpha", args.lora_alpha if args.lora_alpha is not None else args.lora_rank))
    args.lora_target_filter = ",".join(cfg.get("lora_target_filter", parse_name_filter(args.lora_target_filter)))
    args.gate_init = float(cfg.get("gate_init", args.gate_init))


def run_loss(model, pipe, batch, args):
    return anyflow_ltx2_stage1_loss(
        model,
        video_latents=batch["video_latents"],
        audio_latents=batch["audio_latents"],
        video_context=batch["video_context"],
        audio_context=batch["audio_context"],
        video_positions=batch["video_positions"],
        audio_positions=batch["audio_positions"],
        video_patchifier=getattr(pipe, "video_patchifier", None) if pipe is not None else None,
        audio_patchifier=getattr(pipe, "audio_patchifier", None) if pipe is not None else None,
        audio_loss_weight=args.audio_loss_weight,
        boundary_prob=args.boundary_prob,
        fd_eps=args.fd_eps,
        cfg_fused=args.cfg_fused,
        cfg_scale=args.cfg_scale,
        negative_video_context=batch.get("negative_video_context"),
        negative_audio_context=batch.get("negative_audio_context"),
        use_time_weight=not args.disable_time_weight,
        use_adaptive_weight=not args.disable_adaptive_weight,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path")
    parser.add_argument("--dataset_metadata")
    parser.add_argument("--latent_cache_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="a dog running on the grass, natural sound")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=float, default=24)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_target_filter", default="attn,ff,proj")
    parser.add_argument("--allow_tiny_trainable_params", action="store_true")
    parser.add_argument("--gate_init", type=float, default=0.25)
    parser.add_argument("--audio_loss_weight", type=float, default=1.0)
    parser.add_argument("--cfg_fused", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--boundary_prob", type=float, default=0.5)
    parser.add_argument("--fd_eps", type=float, default=1e-3)
    parser.add_argument("--disable_time_weight", action="store_true")
    parser.add_argument("--disable_adaptive_weight", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_test_real_wrapper", action="store_true")
    parser.add_argument("--allow_trainable_without_grad", action="store_true")
    parser.add_argument("--save_gradient_sanity", action="store_true", default=True)
    args = parser.parse_args()
    args._gradient_sanity_checked = False
    apply_resume_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_arg(args.dtype) if device.type == "cuda" else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    if args.smoke_test:
        model = TinyAnyFlowModel().to(device=device, dtype=dtype)
        pipe = None
        loader = [None] * args.max_steps
    else:
        if not args.use_lora and not args.allow_tiny_trainable_params:
            raise RuntimeError("Real Stage 1 training requires --use_lora unless --allow_tiny_trainable_params is set.")
        pipe, model = build_real_model(args, device, dtype)
        model.to(device=device, dtype=dtype)
        if args.smoke_test_real_wrapper:
            loader = [None] * args.max_steps
        else:
            if args.dataset_metadata is None:
                raise ValueError("--dataset_metadata is required unless --smoke_test or --smoke_test_real_wrapper is set")
            dataset = LatentCacheDataset(args.dataset_metadata, args.latent_cache_dir)
            loader = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate))

    print_trainable_report(
        model,
        allow_tiny=args.allow_tiny_trainable_params,
        tiny_ok=args.smoke_test,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    start_step = load_checkpoint(model, optimizer, args.resume) if args.resume else 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.max_steps + 1):
        if args.smoke_test:
            batch = {
                "video_latents": torch.randn(args.batch_size, 4, 3, 8, 8),
                "audio_latents": torch.randn(args.batch_size, 4, 16, 8),
                "video_context": torch.randn(args.batch_size, 2, 8),
                "audio_context": torch.randn(args.batch_size, 2, 8),
                "video_positions": torch.zeros(args.batch_size, 3, 3, 8, 8),
                "audio_positions": torch.zeros(args.batch_size, 1, 16, 2),
            }
        elif args.smoke_test_real_wrapper:
            inputs_shared, inputs_posi = prepare_pipeline_inputs(pipe, args, device)
            batch = {
                "video_latents": inputs_shared["video_latents"].detach(),
                "audio_latents": inputs_shared["audio_latents"].detach(),
                "video_positions": inputs_shared["video_positions"],
                "audio_positions": inputs_shared["audio_positions"],
                "video_context": inputs_posi["video_context"],
                "audio_context": inputs_posi["audio_context"],
            }
            if args.cfg_fused:
                batch["negative_video_context"] = inputs_posi["negative_video_context"]
                batch["negative_audio_context"] = inputs_posi["negative_audio_context"]
        else:
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate))
                batch = next(loader)
            batch = ensure_contexts(pipe, batch, args.cfg_fused)

        batch = move_batch(batch, device, dtype)
        loss, logs = run_loss(model, pipe, batch, args)
        (loss / args.grad_accum_steps).backward()
        if not args._gradient_sanity_checked:
            run_gradient_sanity_or_raise(model, args, step)
        if step % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0 or args.smoke_test or args.smoke_test_real_wrapper:
            msg = {k: float(v.detach().cpu()) for k, v in logs.items()}
            print(f"step={step} {msg}", flush=True)
        if step % args.save_steps == 0 or ((args.smoke_test or args.smoke_test_real_wrapper) and step == args.max_steps):
            ckpt = save_checkpoint(args.output_dir, step, model, optimizer, args)
            if args.smoke_test_real_wrapper:
                reload_pipe, reload_model = build_real_model(args, device, dtype)
                reload_model.to(device=device, dtype=dtype)
                state = torch.load(ckpt / "anyflow_wrapper.pt", map_location="cpu")
                load_trainable_state_dict(reload_model, state, strict_trainable=True)
                reload_model.eval()
                with torch.no_grad():
                    _ = reload_model(
                        video_latents=batch["video_latents"],
                        audio_latents=batch["audio_latents"],
                        video_context=batch["video_context"],
                        audio_context=batch["audio_context"],
                        video_positions=batch["video_positions"],
                        audio_positions=batch["audio_positions"],
                        timestep=torch.ones(batch["video_latents"].shape[0], device=device, dtype=dtype),
                        r_timestep=torch.zeros(batch["video_latents"].shape[0], device=device, dtype=dtype),
                        video_patchifier=reload_pipe.video_patchifier,
                        audio_patchifier=reload_pipe.audio_patchifier,
                    )
                print(f"real wrapper checkpoint reload verified: {ckpt}", flush=True)


if __name__ == "__main__":
    main()
