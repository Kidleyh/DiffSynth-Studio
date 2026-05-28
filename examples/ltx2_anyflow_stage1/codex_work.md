# Codex Work Log

## 2026-05-27 - Harden Stage 1 Train/Save/Sample Path

- Split minimal LoRA utilities into `anyflow_ltx2_lora.py` so training and sampling use identical injection config.
- Fixed copied `r_adaln` parameters so they remain trainable after the base DiT is frozen.
- Reworked Stage 1 loss to separate `time_weight` from AnyFlow adaptive reweighting.
- Replaced CFG sampling-style formula with AnyFlow guidance-fused training formula.
- Added robust checkpoint config save/load, resume model weight loading, and strict checks for critical missing/unexpected keys.
- Added latent-only rollout sampling that saves final latents and per-step rollout stats without decoding.
- Added tiny and real-wrapper smoke test entry points.
- Updated README commands and made LoRA rank 256 the recommended formal training path.

## Round 3: Gradient sanity and unused r_adaln cleanup

### A. Modified Files

- `examples/ltx2_anyflow_stage1/README.md`
- `examples/ltx2_anyflow_stage1/anyflow_ltx2_debug.py`
- `examples/ltx2_anyflow_stage1/anyflow_ltx2_model_wrapper.py`
- `examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1.py`
- `examples/ltx2_anyflow_stage1/codex_work.md`

### B. Scope

No files outside examples/ltx2_anyflow_stage1 were modified.

### C. r_adaln.linear Handling

`r_adaln.linear` is frozen with `requires_grad=False`. The current forward path uses `r_adaln.emb(r)` for r-timestep conditioning, blends that embedding with `base_adaln.emb(t)`, and then projects with `base_adaln.linear(...)`. Because `r_adaln.linear` does not participate in this forward formula, leaving it trainable would inflate trainable parameter counts and save unused checkpoint tensors.

The r-timestep related trainable parameters that remain enabled are `r_adaln.emb.*` and `gate`; LoRA parameters remain trainable when injected.

### D. Gradient Sanity Check

Implemented in `examples/ltx2_anyflow_stage1/anyflow_ltx2_debug.py` as `collect_gradient_sanity(model)`.

Returned fields:

- `total_trainable_tensors`
- `trainable_with_grad_count`
- `trainable_without_grad_count`
- `trainable_with_nonzero_grad_count`
- `trainable_zero_grad_count`
- `trainable_without_grad_names`
- `trainable_zero_grad_names`
- `grad_norm_top20`
- `trainable_param_count_by_prefix`

The trainer calls this after the first backward pass and before `optimizer.step()`, prints the report, saves `gradient_sanity_step_000001.json` by default, raises on trainable tensors without gradients unless `--allow_trainable_without_grad` is passed, and warns on zero gradients.

### E. Tiny Smoke Test

Command:

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

Result: passed.

`gradient_sanity_step_000001.json` summary:

- `total_trainable_tensors`: 6
- `trainable_with_grad_count`: 6
- `trainable_without_grad_count`: 0
- `trainable_with_nonzero_grad_count`: 6
- `trainable_zero_grad_count`: 0
- `trainable_without_grad_names` first 20: `[]`
- `trainable_zero_grad_names` first 20: `[]`
- `grad_norm_top20` first 20:
  - `["time.bias", 0.7461655735969543]`
  - `["audio.weight", 0.5961034893989563]`
  - `["video.bias", 0.5617402195930481]`
  - `["video.weight", 0.5209435224533081]`
  - `["audio.bias", 0.49113234877586365]`
  - `["time.weight", 0.17138135433197021]`

### F. Tiny Checkpoint Resume

Command:

```bash
python examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1.py \
  --smoke_test \
  --output_dir outputs/ltx2_anyflow_stage1_tiny_smoke \
  --batch_size 1 \
  --max_steps 3 \
  --save_steps 1 \
  --dtype fp32 \
  --use_lora \
  --lora_rank 8 \
  --save_gradient_sanity \
  --resume outputs/ltx2_anyflow_stage1_tiny_smoke/checkpoint-step_000002
```

Result: passed.

Load summary: `missing=0 unexpected=0 global_step=2`.

### G. py_compile

Command:

```bash
python -m py_compile examples/ltx2_anyflow_stage1/*.py
```

Result: passed.

### H. Real Wrapper Smoke Test

Not run. Reason: no valid real `model_config_path` was provided in this turn, and the repository did not contain a ready-to-use `model_config.json` path for LTX2 weights.

### I. Known Limits

- Stage 1 only; no DMD, no real_score/fake_score/discriminator, and no on-policy distillation.
- Real LTX2 wrapper still requires a valid `model_config_path` and real model weights to verify the full shape/context/positions path.
- The tiny smoke test validates trainer, loss, checkpoint, resume, and gradient sanity mechanics, but does not validate real LTX2 tensor shapes.

## Round 4: Native LTX2 cache/training integration

### A. Files changed

- `examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py`
- `examples/ltx2_anyflow_stage1/anyflow_ltx2_native_loss_adapter.py`
- `examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh`
- `examples/ltx2_anyflow_stage1/anyflow_ltx2_stage1_loss.py`
- `examples/ltx2_anyflow_stage1/anyflow_ltx2_model_wrapper.py`
- `examples/ltx2_anyflow_stage1/anyflow_ltx2_debug.py`
- `examples/ltx2_anyflow_stage1/README.md`
- `examples/ltx2_anyflow_stage1/codex_work.md`

### B. Scope

No files outside examples/ltx2_anyflow_stage1 were modified.

### C. Native trainer reuse

`train_ltx2_anyflow_stage1_native.py` mirrors the native `examples/ltx2/model_training/train.py` structure: it uses `ModelConfig`, `LTX2AudioVideoPipeline.from_pretrained`, `UnifiedDataset`, native data operators, `launch_data_process_task`, `ModelLogger`, `Accelerator`, and the native DeepSpeed/FSDP-style parser arguments.

`anyflow_stage1:data_process` delegates to the native data-process launcher and produces the same cache style as `sft:data_process`.

`anyflow_stage1:train` reads the native cache, runs the same pipeline units to produce `inputs_shared` / `inputs_posi`, wraps `pipe.dit` with `LTX2AnyFlowWrapper`, injects LoRA into the DiT, and replaces only the final `FlowMatchSFTAudioVideoLoss` call with `anyflow_stage1_native_loss(...)`.

### D. Native cache compatibility

No AnyFlow-specific cache format is introduced.

The native loss adapter reads:

- `inputs_shared["input_latents"]`
- optional `inputs_shared["audio_input_latents"]`
- `inputs_shared["video_positions"]`
- `inputs_shared["audio_positions"]`
- `inputs_posi["video_context"]`
- `inputs_posi["audio_context"]`
- optional negative contexts for CFG-fused training
- patchifiers and gradient-checkpoint flags from `inputs_shared` / `pipe`

Missing required fields raise a clear `KeyError` with the available keys. The smoke metadata used in validation could not load audio from the listed mp4 files, so `audio_input_latents` was absent; the adapter supports this video-only native-cache fallback while preserving the audio branch when audio latents are present.

### E. LoRA alignment

The native trainer supports the native LoRA argument style:

- `--lora_base_model dit`
- `--lora_target_modules "to_k,to_q,to_v,to_out.0"`
- `--lora_rank`
- `--lora_alpha`
- `--learning_rate`
- `--weight_decay`

Default target modules are `to_k,to_q,to_v,to_out.0`. The smoke script uses rank 8 and alpha 8. Formal training should use rank 256 and alpha 256.

### F. Checkpoint and config

Native sidecar checkpoints write:

- `checkpoint-step_000001/anyflow_wrapper.pt`
- `checkpoint-step_000001/optimizer.pt`
- `checkpoint-step_000001/training_state.json`
- `checkpoint-step_000001/anyflow_config.json`
- `gradient_sanity_step_000001.json`

`anyflow_config.json` records `native_trainer`, task name, model path strings, LoRA base/targets/rank/alpha, gate init, loss settings, CFG settings, time-weight/adaptive-loss settings, gradient sanity state, and the frozen unused `r_adaln.linear` flag.

For DeepSpeed ZeRO native runs, `anyflow_wrapper.pt` gathers only trainable AnyFlow/LoRA parameters with ZeRO-safe helpers. The sidecar `optimizer.pt` stores lightweight metadata instead of gathering the full ZeRO optimizer state; full optimizer recovery should use the native DeepSpeed/Accelerate checkpoint stack.

### G. Gradient sanity check

Gradient sanity is connected after the first `accelerator.backward(loss)` and before `optimizer.step()`. It saves `output_path/gradient_sanity_step_000001.json`.

The check is ZeRO-aware: `collect_gradient_sanity` uses DeepSpeed `safe_get_full_grad(param)` when ordinary `param.grad` is partitioned away. If trainable tensors have no gradient, the trainer raises by default; `--allow_trainable_without_grad` can allow debugging runs.

### H. Actual validation

`py_compile` command:

```bash
python -m py_compile examples/ltx2_anyflow_stage1/*.py
```

Result: passed in the remote `py312` environment.

Native smoke shell command used for a resource-bounded validation:

```bash
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-lora-smoke4-codex-small \
HEIGHT=128 WIDTH=128 NUM_FRAMES=9 \
MAX_STEPS=1 SAVE_STEPS=1 TRAIN_DATASET_REPEAT=1 \
USE_GRADIENT_CHECKPOINTING=0 \
bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh
```

Result: data_process completed and generated native cache. The train stage completed with code 0 after a train-only rerun from the generated cache.

Train-only command used for the final passing save/checkpoint validation:

```bash
accelerate launch --config_file examples/ltx2/model_training/full/accelerate_config_zero3offload_4gpu.yaml \
  examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py \
  --dataset_base_path ./models/train/LTX2.3-I2AV-anyflow-stage1-lora-smoke4-codex-small-cache \
  --data_file_keys "video,input_audio" \
  --extra_inputs "input_audio,input_image" \
  --height 128 --width 128 --num_frames 9 --frame_rate 25 \
  --dataset_num_workers 0 \
  --trainable_models "dit" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors" \
  --initialize_model_on_cpu \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --save_steps 1 \
  --max_steps 1 \
  --output_path ./models/train/LTX2.3-I2AV-anyflow-stage1-lora-smoke4-codex-small \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_q,to_v,to_out.0" \
  --lora_rank 8 \
  --lora_alpha 8 \
  --save_gradient_sanity \
  --task "anyflow_stage1:train"
```

Result: passed.

Output files observed:

- `gradient_sanity_step_000001.json`
- `step-1.safetensors`
- `checkpoint-step_000001/anyflow_wrapper.pt`
- `checkpoint-step_000001/optimizer.pt`
- `checkpoint-step_000001/training_state.json`
- `checkpoint-step_000001/anyflow_config.json`

Gradient sanity summary:

- `total_trainable_tensors`: 2440
- `trainable_with_grad_count`: 2440
- `trainable_without_grad_count`: 0
- `trainable_with_nonzero_grad_count`: 394
- `trainable_zero_grad_count`: 2046
- `grad_norm_top20` first 3:
  - `["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", 11.492180824279785]`
  - `["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", 7.996912956237793]`
  - `["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", 2.9534082412719727]`

The original 512x768x121 smoke configuration was attempted. With LTX2 gradient checkpointing enabled it hit a PyTorch checkpoint recompute metadata mismatch in the native LTX2 transformer path when audio was absent from the smoke cache; with checkpointing disabled at full size it exceeded 80GB H100 memory. The reduced-size native smoke above was used as the final executable verification.

### I. Known limits

- Stage 1 only, no DMD/on-policy.
- Full correctness still requires running the native smoke script with the real dataset/cache/model paths and valid audio-bearing samples.
- The old synthetic trainer remains for unit tests only.
- The smoke metadata currently produced video-only cache because the listed mp4 audio could not be loaded by the native audio operator.
- DeepSpeed ZeRO optimizer recovery is not fully represented by the lightweight AnyFlow sidecar `optimizer.pt`; use the native DeepSpeed/Accelerate checkpoint stack for full optimizer recovery.
