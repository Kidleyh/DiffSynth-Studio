import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from anyflow_ltx2_model_wrapper import LTX2AnyFlowWrapper, trainable_state_dict
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


class LoRALinear(torch.nn.Module):
    def __init__(self, base, rank=16, alpha=None):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = float(alpha or rank)
        self.lora_A = torch.nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, base.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_B.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * (self.alpha / self.rank)


def inject_lora_linear(module, rank=16, name_filter=("attn", "ff", "proj"), prefix=""):
    updated = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        full_hit = any(token in full_name for token in name_filter)
        if isinstance(child, torch.nn.Linear) and full_hit:
            setattr(module, child_name, LoRALinear(child, rank=rank))
            updated += 1
        else:
            updated += inject_lora_linear(child, rank=rank, name_filter=name_filter, prefix=full_name)
    return updated


def save_checkpoint(output_dir, step, model, optimizer, args):
    ckpt = Path(output_dir) / f"checkpoint-step_{step:05d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(trainable_state_dict(model), ckpt / "anyflow_wrapper.pt")
    torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
    (ckpt / "training_state.json").write_text(json.dumps({"step": step, "args": vars(args)}, indent=2))


def build_real_model(args, device, dtype):
    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline

    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=load_model_configs(args.model_config_path),
    )
    wrapper = LTX2AnyFlowWrapper(pipe.dit, freeze_base=True)
    if args.use_lora:
        updated = inject_lora_linear(wrapper.dit, rank=args.lora_rank)
        print(f"injected LoRA into {updated} linear layers", flush=True)
    return pipe, wrapper


def encode_missing_prompts(pipe, batch, device, dtype):
    if "video_context" in batch and "audio_context" in batch:
        return batch
    prompts = batch.get("prompt")
    if prompts is None:
        raise RuntimeError("Missing prompt_embeds/video_context/audio_context and no prompt column was provided.")
    prompt_unit = next(unit for unit in pipe.units if unit.__class__.__name__ == "LTX2AudioVideoUnit_PromptEmbedder")
    video_context, audio_context = [], []
    with torch.no_grad():
        for prompt in prompts:
            vc, ac, _ = prompt_unit.encode_prompt(pipe, prompt)
            video_context.append(vc.squeeze(0).detach().cpu())
            audio_context.append(ac.squeeze(0).detach().cpu())
    batch["video_context"] = torch.stack(video_context)
    batch["audio_context"] = torch.stack(audio_context)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path")
    parser.add_argument("--dataset_metadata")
    parser.add_argument("--latent_cache_dir")
    parser.add_argument("--output_dir", required=True)
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
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--audio_loss_weight", type=float, default=1.0)
    parser.add_argument("--cfg_fused", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--boundary_prob", type=float, default=0.5)
    parser.add_argument("--fd_eps", type=float, default=1e-3)
    parser.add_argument("--resume")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_arg(args.dtype) if device.type == "cuda" else torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    if args.smoke_test:
        model = TinyAnyFlowModel().to(device=device, dtype=dtype)
        pipe = None
        loader = [None] * args.max_steps
    else:
        if args.dataset_metadata is None:
            raise ValueError("--dataset_metadata is required unless --smoke_test is set")
        pipe, model = build_real_model(args, device, dtype)
        model.to(device=device, dtype=dtype)
        dataset = LatentCacheDataset(args.dataset_metadata, args.latent_cache_dir)
        loader = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate))

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params):,}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    start_step = 0
    if args.resume:
        state = torch.load(Path(args.resume) / "optimizer.pt", map_location="cpu")
        optimizer.load_state_dict(state)
        meta = json.loads((Path(args.resume) / "training_state.json").read_text())
        start_step = int(meta.get("step", 0))

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
        else:
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate))
                batch = next(loader)
            batch = encode_missing_prompts(pipe, batch, device, dtype)
        batch = {k: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else v.to(device=device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        loss, logs = anyflow_ltx2_stage1_loss(
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
        )
        (loss / args.grad_accum_steps).backward()
        if step % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0 or args.smoke_test:
            msg = {k: float(v.detach().cpu()) for k, v in logs.items()}
            print(f"step={step} {msg}", flush=True)
        if step % args.save_steps == 0 or (args.smoke_test and step == args.max_steps):
            save_checkpoint(args.output_dir, step, model, optimizer, args)


if __name__ == "__main__":
    main()
