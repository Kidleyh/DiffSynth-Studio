# AnyFlow-LTX2 Stage 1

This directory implements only AnyFlow Stage 1: Forward Flow Map Training for LTX/LTX2. It does not include DMD, real_score, fake_score, discriminator training, or on-policy distillation.

The Stage 1 objective trains a direct flow-map transition:

`z_r = z_t - (t - r) * u_theta(z_t, t, r)`

The implementation wraps DiffSynth's existing LTX2 DiT without modifying DiffSynth source files. The wrapper adds `r_timestep` conditioning by copying the original timestep embedding path and blending:

`time_emb_anyflow = gate * emb(t) + (1 - gate) * emb_r(r)`

## Important Training Note

Stage 1 needs meaningful trainable capacity. The base LTX2 DiT is frozen by default, so real training should use LoRA.

For the legacy latent-cache trainer, pass `--use_lora --lora_rank 256`. For the native trainer, `anyflow_stage1` and `anyflow_stage1:train` now default to DiT LoRA with `lora_rank=256` and `lora_target_modules=to_k,to_q,to_v,to_out.0`; smoke scripts can still override this to rank 8. `anyflow_stage1:data_process` does not install LoRA.

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

## Native LTX2 Training Path

For real LTX2 training, prefer the native-style entrypoint:

`examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py`

The older `train_ltx2_anyflow_stage1.py` remains useful for tiny/synthetic unit tests and checkpoint mechanics, but real data should flow through the same `UnifiedDataset`, `data_process`, cache, and `pipe.unit_runner` path used by `examples/ltx2/model_training/train.py`.

This native trainer is aligned to the user-validated reference script:

`examples/ltx2/model_training/full/LTX-2.3-I2AV-splited_lyh_smoke4_4gpu.sh`

Native data flow:

`UnifiedDataset/cache -> get_pipeline_inputs or cached inputs -> pipe.unit_runner -> inputs_shared/inputs_posi -> AnyFlow Stage 1 native loss adapter -> anyflow_ltx2_stage1_loss`

No AnyFlow-specific cache format is introduced. `anyflow_stage1:data_process` writes the same style of cache as native LTX2 `sft:data_process`; `anyflow_stage1:train` reads that cache and replaces only the final SFT loss with Stage 1 forward flow-map training.

The AnyFlow wrapper mirrors native `model_fn_ltx2` conditioning:

- `input_latents_video` and `denoise_mask_video` are applied to video latents and video timesteps.
- `ref_frames_latents`, `ref_frames_positions`, `in_context_video_latents`, and `in_context_video_positions` are appended to the video token stream when present.
- `input_latents_audio` and `denoise_mask_audio` are applied to audio latents and audio timesteps when present.
- Stage 1 loss supports `denoise_mask_video` and `denoise_mask_audio` as loss masks and logs mask ratios.

When `--cfg_fused` is enabled, conditional context comes from `inputs_posi` and negative/unconditional context comes from `inputs_nega`. With `--cfg_fused` disabled, the loss uses only the conditional forward and does not depend on `inputs_nega`.

Current command examples are written for the available 4-GPU environment. Do not assume 8 GPUs for smoke or formal commands; if global batch needs to be preserved under tighter resources, increase gradient accumulation rather than silently changing the world size. Historical notes may mention 8-GPU probes, but current native examples use 4 GPUs and the existing ZeRO3/offload 4-GPU config.

Native train tasks default to DiT LoRA. Passing `--lora_base_model "dit"` in commands is still shown for readability and compatibility with older scripts, but it is no longer required for `anyflow_stage1:train`. Formal native training should use `--lora_rank 256 --lora_alpha 256`; smoke runs can override to `--lora_rank 8 --lora_alpha 8`.

Commands:

```bash
python -m py_compile examples/ltx2_anyflow_stage1/*.py
```

Native smoke4 shell:

```bash
bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh
```

For a fast validation run on the 4-GPU host, the shell exposes environment knobs:

```bash
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-lora-smoke4-small \
HEIGHT=128 WIDTH=128 NUM_FRAMES=9 \
MAX_STEPS=1 SAVE_STEPS=1 TRAIN_DATASET_REPEAT=1 \
LOW_RES_SMOKE=1 \
bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh
```

One-step low-res native train + sample validation:

```bash
bash examples/ltx2_anyflow_stage1/test_lowres_train1_sample_4gpu.sh
```

The script can also use an explicit sampler config if needed:

```bash
MODEL_CONFIG_PATH=/path/to/model_config.json \
bash examples/ltx2_anyflow_stage1/test_lowres_train1_sample_4gpu.sh
```

By default it generates a temporary sampler model config from the same LTX-2.3 `model_id:origin_file_pattern` specs used by the native training scripts. This validation script uses 4 GPUs for native training, defaults to `128x128x9`, trains only 1 step with LoRA rank 8, checks the native sidecar checkpoint and first-step JSON logs, then runs 4-step `--latent_rollout_only` sampling from `checkpoint-step_000001`. The sampling stage validates native sidecar LoRA injection, strict checkpoint loading, and latent rollout output files.

Gradient-checkpointing policy validation:

```bash
USE_GRADIENT_CHECKPOINTING=1 \
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-test \
bash examples/ltx2_anyflow_stage1/test_lowres_gc_train1_4gpu.sh
```

When `USE_GRADIENT_CHECKPOINTING=1`, AnyFlow Stage 1 only lets the main conditional training forward inherit checkpointing. Finite-difference target forwards and guidance-fused unconditional forwards are forced through `torch.no_grad()` with `use_gradient_checkpointing=False` and `use_gradient_checkpointing_offload=False`. The first-step JSON log records `gradient_checkpointing_main_forward`, `gradient_checkpointing_target_forward`, and `gradient_checkpointing_uncond_forward` to make this policy auditable. This avoids PyTorch checkpoint recompute metadata mismatches caused by AnyFlow's multiple DiT forwards under ZeRO3.

Check the first-step JSON diagnostics:

```bash
cat models/train/LTX2.3-I2AV-anyflow-stage1-lora-smoke4/native_key_report_step_000001.json
cat models/train/LTX2.3-I2AV-anyflow-stage1-lora-smoke4/anyflow_stage1_log_step_000001.json
cat models/train/LTX2.3-I2AV-anyflow-stage1-lora-smoke4/gradient_sanity_step_000001.json
```

The native key report records the exact target/context keys and which optional condition keys were actually forwarded into the wrapper, including `video_condition_keys_used_in_forward`, `audio_condition_keys_used_in_forward`, `ref_condition_keys_used_in_forward`, `audio_target_key`, and `audio_fallback_reason`.

Train only from an existing native cache with the shell wrapper:

```bash
TRAIN_ONLY=1 \
CACHE_DIR=/path/to/existing-native-cache \
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-lora-smoke4-train-only \
bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh
```

Optional CFG-fused smoke mode uses negative context from `inputs_nega`:

```bash
TRAIN_ONLY=1 \
CACHE_DIR=/path/to/existing-native-cache \
CFG_FUSED=1 CFG_SCALE=1.0 \
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-lora-smoke4-cfg \
bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh
```

Data-process only:

```bash
accelerate launch --num_processes 1 examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py \
  --dataset_base_path "" \
  --dataset_metadata_path examples/ltx2/model_training/full/metadata_lyh_smoke4.csv \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image" \
  --height 512 --width 768 --num_frames 121 --frame_rate 25 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors" \
  --output_path ./models/train/LTX2.3-I2AV-anyflow-stage1-lora-smoke4-cache \
  --task "anyflow_stage1:data_process"
```

Train only from native cache:

```bash
accelerate launch --config_file examples/ltx2/model_training/full/accelerate_config_zero3offload_4gpu.yaml \
  examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py \
  --dataset_base_path ./models/train/LTX2.3-I2AV-anyflow-stage1-lora-smoke4-cache \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image" \
  --height 512 --width 768 --num_frames 121 --frame_rate 25 \
  --dataset_repeat 25 \
  --model_id_with_origin_paths "DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors" \
  --initialize_model_on_cpu \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --save_steps 2 \
  --max_steps 2 \
  --output_path ./models/train/LTX2.3-I2AV-anyflow-stage1-lora-smoke4 \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_q,to_v,to_out.0" \
  --lora_rank 8 \
  --lora_alpha 8 \
  --save_gradient_sanity \
  --task "anyflow_stage1:train"
```

For formal training, change smoke LoRA settings to `--lora_rank 256 --lora_alpha 256` and increase training duration. The train task defaults to DiT LoRA, but explicit `--lora_base_model "dit"` is harmless and documents intent. This remains Stage 1 only: no DMD and no on-policy distillation.

Native sidecar checkpoint sampling example, using a single sampling process from a 4-GPU-trained checkpoint:

```bash
python examples/ltx2_anyflow_stage1/sample_ltx2_anyflow_stage1.py   --checkpoint /path/to/native/checkpoint-step_000001   --model_config_path /path/to/model_config.json   --prompt "a dog running on the grass, natural sound"   --num_inference_steps 4   --latent_rollout_only   --output_path outputs/native_checkpoint_sampling_smoke_4step
```

## Gradient Checkpointing Diagnostic Matrix

Codex should not directly run these GPU diagnostics unless explicitly asked. Run the matrix manually, then ask Codex to read the logs.

A. 4GPU ZeRO3 + gradient checkpointing control:

```bash
USE_GRADIENT_CHECKPOINTING=1 \
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-4gpu-zero3 \
bash examples/ltx2_anyflow_stage1/test_lowres_gc_train1_4gpu.sh
```

Train log:

```bash
./models/train/LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-4gpu-zero3-logs/train.log
```

B. 1GPU no-DeepSpeed + gradient checkpointing localization:

```bash
USE_GRADIENT_CHECKPOINTING=1 \
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-1gpu-nodeepspeed \
bash examples/ltx2_anyflow_stage1/test_lowres_gc_train1_1gpu_nodeepspeed.sh
```

Train log:

```bash
./models/train/LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-1gpu-nodeepspeed-logs/train.log
```

C. 4GPU ZeRO2/offload + gradient checkpointing diagnostic:

```bash
USE_GRADIENT_CHECKPOINTING=1 \
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-4gpu-zero2 \
bash examples/ltx2_anyflow_stage1/test_lowres_gc_train1_4gpu_zero2.sh
```

Train log:

```bash
./models/train/LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-4gpu-zero2-logs/train.log
```

Summarize any run:

```bash
python examples/ltx2_anyflow_stage1/summarize_gc_diagnostic_logs.py \
  --log_dir ./models/train/LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-1gpu-nodeepspeed-logs \
  --output_dir ./models/train/LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-1gpu-nodeepspeed
```

Interpretation:

- If 1GPU no-DeepSpeed passes, ZeRO2 + GC passes, and ZeRO3 + GC fails, the issue is effectively isolated to ZeRO3 parameter partitioning plus PyTorch checkpoint recompute.
- If 1GPU no-DeepSpeed passes but ZeRO2 + GC also fails, the issue is likely broader multi-GPU DeepSpeed plus PyTorch checkpointing, not only ZeRO3.
- If 1GPU no-DeepSpeed also fails, the issue is still in the LTX2 main forward / AnyFlow wrapper checkpoint path.
- If 1GPU OOMs, record the OOM and design a smaller diagnostic case before changing DeepSpeed config.

## Checkpoints

Each checkpoint directory contains:

- `anyflow_wrapper.pt`: trainable AnyFlow and LoRA weights.
- `optimizer.pt`: optimizer state.
- In native DeepSpeed/ZeRO runs this sidecar records lightweight optimizer metadata; full ZeRO optimizer recovery should use the native training/checkpoint stack.
- `training_state.json`: global step and CLI args.
- `anyflow_config.json`: wrapper, LoRA, loss, CFG, and scheduler-relevant config.

Resume restores wrapper/LoRA weights, optimizer, and global step. Sampling reads `anyflow_config.json`, reads `anyflow_wrapper.pt` once, injects LoRA if needed, then strict-loads the already-read state.

Sampler LoRA reconstruction is compatible with native sidecar checkpoints and legacy latent-cache checkpoints. It enables LoRA when any of these are true: `use_lora=true`, `lora_base_model="dit"`, or the checkpoint state dict contains `lora_A`/`lora_B` keys. Target modules prefer native `lora_target_modules`, then legacy `lora_target_filter`, then default to `to_k,to_q,to_v,to_out.0`.

If `anyflow_config.json` does not contain `lora_rank`, the sampler can infer rank from `anyflow_wrapper.pt` by reading `lora_A`/`lora_B` parameter shapes. This keeps old rank-8 smoke/native sidecar checkpoints loadable instead of incorrectly rebuilding rank-256 LoRA. New checkpoints should still rely on explicit `lora_rank` and `lora_alpha` in `anyflow_config.json`.

Native `anyflow_config.json` writes both native fields (`lora_base_model`, `lora_target_modules`) and legacy aliases (`use_lora`, `lora_target_filter`) so older and newer samplers can rebuild the same wrapper structure.

Large numbers of frozen base-model keys missing from `anyflow_wrapper.pt` are normal because checkpoints save only trainable parameters. Missing trainable keys or critical unexpected AnyFlow/LoRA keys raise during load.

`anyflow_config.json` also records gradient-health metadata:

- `gradient_sanity_checked`: whether the run completed the first backward sanity check before saving.
- `frozen_unused_r_adaln_linear`: true when unused copied `r_adaln.linear` parameters are frozen.
- `trainable_without_grad_policy`: `raise` or `allow`, based on `--allow_trainable_without_grad`.

## Gradient Sanity Check

Gradient sanity checks confirm that LoRA, the r-timestep embedding path, and the `gate` parameter actually participate in backward. The first backward pass reports trainable tensors with missing gradients, zero gradients, and the top gradient norms. It writes `gradient_sanity_step_000001.json` by default.

```bash
python examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1.py \
  --smoke_test \
  --output_dir outputs/ltx2_anyflow_stage1_tiny_smoke \
  --batch_size 1 \
  --max_steps 2 \
  --save_steps 2 \
  --dtype fp32 \
  --use_lora \
  --lora_rank 8 \
  --save_gradient_sanity
```

If `trainable_without_grad_names` is non-empty, training raises by default. Pass `--allow_trainable_without_grad` only for debugging.
