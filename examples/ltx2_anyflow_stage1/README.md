# AnyFlow-LTX2 Stage 1

This directory implements only AnyFlow Stage 1: Forward Flow Map Training for LTX/LTX2. It does not include DMD, real_score, fake_score, discriminator training, or on-policy distillation.

The Stage 1 objective trains a direct flow-map transition:

`z_r = z_t - (t - r) * u_theta(z_t, t, r)`

The implementation wraps DiffSynth's existing LTX2 DiT without modifying DiffSynth source files. The wrapper adds `r_timestep` conditioning by copying the original timestep embedding path and blending:

`time_emb_anyflow = gate * emb(t) + (1 - gate) * emb_r(r)`

## Important Training Note

Stage 1 needs meaningful trainable capacity. The base LTX2 DiT is frozen by default, so real training should use LoRA. Recommended setting:

`--use_lora --lora_rank 256`

If real training has fewer than 1M trainable parameters, the trainer raises unless `--allow_tiny_trainable_params` is explicitly passed for debugging.

## Files

- `anyflow_ltx2_scheduler.py`: normalized continuous-time Euler scheduler for noise injection and any-step rollout.
- `anyflow_ltx2_model_wrapper.py`: non-invasive LTX2 DiT wrapper with copied r-time embedding.
- `anyflow_ltx2_lora.py`: shared minimal LoRA injection and trainable-parameter reporting utilities.
- `anyflow_ltx2_stage1_loss.py`: Stage 1 central-difference target, time weighting, adaptive weighting, and optional guidance-fused formula.
- `train_ltx2_anyflow_stage1.py`: latent-cache trainer, tiny smoke test, real-wrapper smoke test, checkpoint save/resume.
- `sample_ltx2_anyflow_stage1.py`: latent-only rollout and experimental full decode sampling.
- `codex_work.md`: running implementation notes for future Codex edits.

## Loss Formula

For clean latent `x` and noise `eps`:

`z_t = (1 - t) * x + t * eps`

`v = eps - x`

The wrapper predicts `u_theta(z_t, t, r)`. The central-difference target is:

`du_dt ~= [u_theta(z_t + eps_fd * v, t + eps_fd, r) - u_theta(z_t - eps_fd * v, t - eps_fd, r)] / (2 * eps_fd)`

`u_tgt = v - (t - r) * du_dt`

`time_weight` is the optional timestep prior `w(t)` using Beta(2, 1.5)-shaped weighting. `adaptive_weight` is the AnyFlow-style per-sample reweighting: boundary samples use 1, non-boundary samples use `mu_boundary / (reg_total.detach() + c)`. If a batch has no boundary sample, adaptive weighting falls back to 1 and logs `adaptive_fallback=True`.

Guidance-fused training, when enabled, uses:

`u = (u_cond - (1 - g) * stopgrad(u_uncond)) / g`

Early Stage 1 runs should first disable `--cfg_fused`; enabling it requires empty-prompt/unconditional contexts.

## Data Format

The trainer expects latent-cache metadata as CSV or JSONL. Each row should contain:

- `video_latents`: path to a `.pt` video latent tensor.
- `audio_latents`: path to a `.pt` audio latent tensor.
- `video_positions`: path to LTX2 video positions tensor.
- `audio_positions`: path to LTX2 audio positions tensor.
- `prompt` or cached context tensors.
- `prompt_embeds`, or separate `video_context` and `audio_context`.
- Optional `negative_video_context` and `negative_audio_context` for `--cfg_fused`.

Relative paths are resolved under `--latent_cache_dir`.

## Commands

### 1. py_compile

```bash
python -m py_compile examples/ltx2_anyflow_stage1/*.py
```

### 2. Tiny Smoke Test

```bash
python examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1.py \
  --smoke_test \
  --output_dir outputs/ltx2_anyflow_stage1_tiny_smoke \
  --batch_size 1 \
  --max_steps 2 \
  --save_steps 2 \
  --dtype fp32 \
  --use_lora \
  --lora_rank 8
```

### 3. Real Wrapper Smoke Test

```bash
python examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1.py \
  --smoke_test_real_wrapper \
  --model_config_path /path/to/model_config.json \
  --output_dir outputs/ltx2_anyflow_stage1_real_smoke \
  --batch_size 1 \
  --max_steps 2 \
  --save_steps 2 \
  --dtype bf16 \
  --use_lora \
  --lora_rank 8
```

### 4. Latent-Only 4/8/16/32 Rollout

```bash
for steps in 4 8 16 32; do
  python examples/ltx2_anyflow_stage1/sample_ltx2_anyflow_stage1.py \
    --checkpoint outputs/ltx2_anyflow_stage1_real_smoke/checkpoint-step_000002 \
    --model_config_path /path/to/model_config.json \
    --prompt "a dog running on the grass, natural sound" \
    --num_inference_steps ${steps} \
    --latent_rollout_only \
    --output_path outputs/ltx2_anyflow_stage1_real_smoke/latent_rollout_${steps}step
done
```

### 5. Formal Latent-Cache Training

```bash
python examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1.py \
  --model_config_path /path/to/model_config.json \
  --dataset_metadata /path/to/metadata.csv \
  --latent_cache_dir /path/to/latent_cache \
  --output_dir outputs/ltx2_anyflow_stage1_train \
  --height 512 --width 768 --num_frames 121 --frame_rate 24 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --learning_rate 1e-5 \
  --max_steps 10000 \
  --save_steps 500 \
  --dtype bf16 \
  --use_lora \
  --lora_rank 256 \
  --lora_alpha 256 \
  --audio_loss_weight 1.0 \
  --boundary_prob 0.5 \
  --fd_eps 0.001
```

## Experimental Full Decode Sampling

Full decode still depends on the original LTX2 VAE/vocoder pipeline and is intentionally secondary to latent-only validation:

```bash
python examples/ltx2_anyflow_stage1/sample_ltx2_anyflow_stage1.py \
  --checkpoint outputs/ltx2_anyflow_stage1_train/checkpoint-step_010000 \
  --model_config_path /path/to/model_config.json \
  --prompt "a dog running on the grass, natural sound" \
  --num_inference_steps 8 \
  --output_path outputs/ltx2_anyflow_stage1_train/sample_8step.mp4
```

## Checkpoints

Each checkpoint directory contains:

- `anyflow_wrapper.pt`: trainable AnyFlow and LoRA weights.
- `optimizer.pt`: optimizer state.
- `training_state.json`: global step and CLI args.
- `anyflow_config.json`: wrapper, LoRA, loss, CFG, and scheduler-relevant config.

Resume restores wrapper/LoRA weights, optimizer, and global step. Sampling reads `anyflow_config.json`, injects LoRA if needed, then loads `anyflow_wrapper.pt`.

