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

## Round 5: I2AV conditioning and native smoke alignment

### A. Files changed

- `examples/ltx2_anyflow_stage1/anyflow_ltx2_model_wrapper.py`
- `examples/ltx2_anyflow_stage1/anyflow_ltx2_native_loss_adapter.py`
- `examples/ltx2_anyflow_stage1/anyflow_ltx2_stage1_loss.py`
- `examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py`
- `examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh`
- `examples/ltx2_anyflow_stage1/README.md`
- `examples/ltx2_anyflow_stage1/codex_work.md`

### B. Scope

No files outside examples/ltx2_anyflow_stage1 were modified.

### C. Reference script alignment

The alignment reference is `examples/ltx2/model_training/full/LTX-2.3-I2AV-splited_lyh_smoke4_4gpu.sh`, which the user confirmed can run successfully.

The AnyFlow smoke script keeps the same structure and parameter style:

- dataset metadata: `examples/ltx2/model_training/full/metadata_lyh_smoke4.csv`
- `--data_file_keys "video,input_audio"`
- `--extra_inputs "input_audio,input_image"`
- two stages: `anyflow_stage1:data_process` then `anyflow_stage1:train`
- encoder model spec: text encoder post modules, video VAE encoder, audio VAE encoder, Gemma text encoder
- train model spec: `DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors`
- 4-GPU ZeRO-3 accelerate config path: `examples/ltx2/model_training/full/accelerate_config_zero3offload_4gpu.yaml`

The train stage adds AnyFlow LoRA parameters: `--lora_base_model dit`, `--lora_target_modules "to_k,to_q,to_v,to_out.0"`, `--lora_rank 8`, `--lora_alpha 8`, and `--save_gradient_sanity`. Formal training should use rank/alpha 256.

### D. Conditioning support

The wrapper now implements an AnyFlow version of native `model_fn_ltx2` and preserves I2AV conditioning for all Stage 1 forward calls.

Required keys:

- `inputs_shared.input_latents`
- `inputs_shared.video_positions`
- `inputs_shared.audio_positions`
- `inputs_posi.video_context`
- `inputs_posi.audio_context`

Video optional condition keys:

- `input_latents_video`
- `denoise_mask_video`
- `ref_frames_latents`
- `ref_frames_positions`
- `in_context_video_latents`
- `in_context_video_positions`

Audio optional condition keys:

- `input_latents_audio`
- `denoise_mask_audio`

Ref/in-context keys:

- `ref_frames_latents`
- `ref_frames_positions`
- `in_context_video_latents`
- `in_context_video_positions`

`denoise_mask_video` and `denoise_mask_audio` are supported. If optional keys are missing, they are passed as `None`. If required keys are missing, the adapter raises a clear error and prints available `inputs_shared`, `inputs_posi`, and `inputs_nega` keys.

### E. CFG fused correction

Conditional context is read from `inputs_posi`. Negative/unconditional context is read from `inputs_nega`.

When `cfg_fused=False`, AnyFlow Stage 1 uses only the conditional forward and does not depend on `inputs_nega`.

When `cfg_fused=True`, the adapter requires `inputs_nega["video_context"]` and `inputs_nega["audio_context"]`. Missing negative context raises a clear error with available keys. The formula remains:

```text
u = (u_cond - (1 - g) * stopgrad(u_uncond)) / g
```

### F. Loss mask correction

AnyFlow Stage 1 loss now supports video/audio loss masks.

Mask sources:

- video: `denoise_mask_video`
- audio: `denoise_mask_audio`

The per-sample regression loss applies the mask before reduction. If no mask is present, the loss falls back to full-token supervision. Logs include:

- `video_loss_mask_ratio`
- `audio_loss_mask_ratio`
- `using_video_loss_mask`
- `using_audio_loss_mask`

### G. Validation results

`py_compile` command:

```bash
python -m py_compile examples/ltx2_anyflow_stage1/*.py
```

Result: passed in the remote `py312` environment.

Full smoke command:

```bash
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-r5-full-codex \
MAX_STEPS=1 SAVE_STEPS=1 TRAIN_DATASET_REPEAT=1 \
bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh
```

Result: data_process completed. Train failed during backward with PyTorch checkpoint recompute metadata mismatch:

```text
torch.utils.checkpoint.CheckpointError: Recomputed values for the following tensors have different metadata than during the forward pass.
saved metadata: {'shape': torch.Size([4096]), 'dtype': torch.bfloat16, ...}
recomputed metadata: {'shape': torch.Size([0]), 'dtype': torch.bfloat16, ...}
```

Low-res smoke command:

```bash
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-r5-lowres-codex \
LOW_RES_SMOKE=1 MAX_STEPS=1 SAVE_STEPS=1 TRAIN_DATASET_REPEAT=1 \
bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh
```

Result: passed.

Final train-only rerun from the low-res cache also passed after the `using_negative_context` report correction:

```bash
accelerate launch --config_file examples/ltx2/model_training/full/accelerate_config_zero3offload_4gpu.yaml \
  examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py \
  --dataset_base_path ./models/train/LTX2.3-I2AV-anyflow-stage1-r5-lowres-codex-cache \
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
  --output_path ./models/train/LTX2.3-I2AV-anyflow-stage1-r5-lowres-codex \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_q,to_v,to_out.0" \
  --lora_rank 8 \
  --lora_alpha 8 \
  --save_gradient_sanity \
  --task "anyflow_stage1:train"
```

### H. JSON summaries

`native_key_report_step_000001.json` summary:

- `using_video_conditioning`: `true`
- `using_audio_conditioning`: `false`
- `using_ref_frame_conditioning`: `false`
- `negative_context_available`: `true`
- `using_negative_context`: `false`
- `optional_condition_keys_found`: `["input_latents_video", "denoise_mask_video"]`
- `optional_condition_keys_missing`: `["ref_frames_latents", "ref_frames_positions", "in_context_video_latents", "in_context_video_positions", "input_latents_audio", "denoise_mask_audio"]`

`anyflow_stage1_log_step_000001.json` summary:

- `loss_total`: `0.645950973033905`
- `loss_video`: `0.645950973033905`
- `loss_audio`: `0.0`
- `audio_present`: `false`
- `audio_fallback_reason`: `audio_input_latents missing from native cache/unit outputs`
- `audio_loss_weight`: `1.0`
- `video_loss_mask_ratio`: `0.5`
- `audio_loss_mask_ratio`: `0.0`
- `using_video_loss_mask`: `1.0`
- `using_audio_loss_mask`: `0.0`
- `using_video_conditioning`: `true`
- `using_audio_conditioning`: `false`
- `using_negative_context`: `false`

`gradient_sanity_step_000001.json` summary:

- `total_trainable_tensors`: `2440`
- `trainable_with_grad_count`: `2440`
- `trainable_without_grad_count`: `0`
- `trainable_with_nonzero_grad_count`: `394`
- `trainable_zero_grad_count`: `2046`
- `trainable_without_grad_names` first 20: `[]`
- `trainable_zero_grad_names` first 20:
  - `pipe.dit.dit.audio_adaln_single.gate`
  - `pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight`
  - `pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias`
  - `pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight`
  - `pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias`
  - `pipe.dit.dit.audio_prompt_adaln_single.gate`
  - `pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight`
  - `pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias`
  - `pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight`
  - `pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias`
  - `pipe.dit.dit.av_ca_video_scale_shift_adaln_single.gate`
  - `pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight`
  - `pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias`
  - `pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight`
  - `pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias`
  - `pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.gate`
  - `pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight`
  - `pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias`
  - `pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight`
  - `pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias`
- `grad_norm_top20` first 20:
  - `["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", 2.79573917388916]`
  - `["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", 2.239431142807007]`
  - `["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", 0.9807363748550415]`
  - `["pipe.dit.dit.adaln_single.gate", 0.75]`
  - `["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", 0.3126341998577118]`
  - `["pipe.dit.dit.prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", 0.12585578858852386]`
  - `["pipe.dit.dit.prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", 0.09677280485630035]`
  - `["pipe.dit.dit.transformer_blocks.36.attn2.to_q.lora_B.weight", 0.03541587293148041]`
  - `["pipe.dit.dit.prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", 0.019645068794488907]`
  - `["pipe.dit.dit.transformer_blocks.0.attn1.to_v.lora_B.weight", 0.01540923397988081]`
  - `["pipe.dit.dit.transformer_blocks.35.attn1.to_v.lora_B.weight", 0.015403569675981998]`
  - `["pipe.dit.dit.transformer_blocks.45.attn2.to_q.lora_B.weight", 0.013852784410119057]`
  - `["pipe.dit.dit.transformer_blocks.35.attn1.to_out.0.lora_B.weight", 0.013552132993936539]`
  - `["pipe.dit.dit.transformer_blocks.46.attn1.to_out.0.lora_B.weight", 0.012469983659684658]`
  - `["pipe.dit.dit.transformer_blocks.36.attn1.to_v.lora_B.weight", 0.012225748039782047]`
  - `["pipe.dit.dit.transformer_blocks.39.attn1.to_out.0.lora_B.weight", 0.011765757575631142]`
  - `["pipe.dit.dit.transformer_blocks.41.attn1.to_v.lora_B.weight", 0.011410781182348728]`
  - `["pipe.dit.dit.prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", 0.011318734847009182]`
  - `["pipe.dit.dit.transformer_blocks.47.attn1.to_k.lora_B.weight", 0.01124351192265749]`
  - `["pipe.dit.dit.transformer_blocks.41.attn1.to_out.0.lora_B.weight", 0.010870256461203098]`

### I. Known limits

- Stage 1 only, no DMD/on-policy.
- Full AnyFlow paper reproduction still requires Stage 2 flow map backward simulation plus DMD.
- Full-resolution AnyFlow smoke currently fails under gradient checkpointing with PyTorch checkpoint recompute metadata mismatch. The current workaround is `LOW_RES_SMOKE=1`, which disables LTX2 gradient checkpointing and uses smaller `128x128x9` latents for executable validation.
- The current smoke metadata cannot load audio from the referenced mp4 files in the native audio operator, so `audio_present=false` and audio-specific gradients are zero in this validation. The wrapper/adapter path supports audio conditioning keys when a native cache contains `audio_input_latents` / `input_latents_audio`.
