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

## Round 6: Real I2AV conditioning verification

### A. Files changed

- `examples/ltx2_anyflow_stage1/anyflow_ltx2_native_loss_adapter.py`
- `examples/ltx2_anyflow_stage1/anyflow_ltx2_stage1_loss.py`
- `examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh`
- `examples/ltx2_anyflow_stage1/README.md`
- `examples/ltx2_anyflow_stage1/codex_work.md`

### B. Files outside this directory

No files outside examples/ltx2_anyflow_stage1 were modified.

### C. Alignment with the user-verified script

- Reference script: `examples/ltx2/model_training/full/LTX-2.3-I2AV-splited_lyh_smoke4_4gpu.sh`.
- Dataset/metadata: default metadata remains `examples/ltx2/model_training/full/metadata_lyh_smoke4.csv`.
- `extra_inputs`: remains `input_audio,input_image`.
- `data_file_keys`: remains `video,input_audio`.
- `model_id_with_origin_paths`: encoder phase uses text post modules, video VAE encoder, audio VAE encoder, and Gemma; train phase uses `DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors`.
- `input_image` and `input_audio` are still kept in the smoke script. `LOW_RES_SMOKE=1` only changes resource knobs (`height`, `width`, `num_frames`, and gradient checkpointing).

### D. Native model_fn_ltx2 condition key table

| Native key | AnyFlow adapter extracts | AnyFlow wrapper forward uses | optional/required | smoke observed |
| --- | --- | --- | --- | --- |
| `input_latents` | yes | yes, as clean video target before noise | required | yes |
| `video_positions` | yes | yes, passed to DiT and extended for refs | required | yes |
| `video_context` | yes from `inputs_posi` | yes, passed to DiT | required | yes |
| `input_latents_video` | yes | yes, patchified and mixed into noisy video latents | optional | yes |
| `denoise_mask_video` | yes | yes, patchified; controls video latent replacement and video timesteps; also used as loss mask | optional, required if `input_latents_video` exists | yes |
| `ref_frames_latents` | yes | yes, appended to video token stream when present | optional | no |
| `ref_frames_positions` | yes | yes, appended with ref latents when present | optional | no |
| `in_context_video_latents` | yes | yes, appended to video token stream when present | optional | no |
| `in_context_video_positions` | yes | yes, appended with in-context latents when present | optional | no |
| `audio_input_latents` | yes | yes, as clean audio target when present | optional target | no |
| `audio_positions` | yes | yes, passed to DiT | required by native adapter | yes |
| `audio_context` | yes from `inputs_posi` | yes, passed to DiT | required | yes |
| `input_latents_audio` | yes | yes, patchified and mixed into noisy audio latents when audio target exists | optional | no |
| `denoise_mask_audio` | yes | yes, patchified; controls audio latent replacement/audio timesteps and loss mask when audio target exists | optional, required if `input_latents_audio` exists | no |
| `inputs_nega.video_context` | yes when CFG fused | yes as unconditional video context | required only for `cfg_fused=True` | yes in CFG smoke |
| `inputs_nega.audio_context` | yes when CFG fused | yes as unconditional audio context | required only for `cfg_fused=True` | yes in CFG smoke |

### E. Video conditioning conclusion

- `using_video_conditioning`: true in low-res and train-only smoke.
- Keys used in forward: `input_latents_video`, `denoise_mask_video`.
- These keys do not just appear in the report: `LTX2AnyFlowWrapper.anyflow_model_fn_ltx2` patchifies the mask, replaces the noisy video latent tokens with `input_latents_video` where the native denoise mask is zero, and applies the same mask to video timesteps. The same `condition_kwargs` are passed to the main prediction, central-difference `u_plus/u_minus`, and CFG-fused uncond forward.

### F. Audio conditioning conclusion

- `audio_present`: false in the available smoke data.
- `audio_target_key`: null because `inputs_shared.audio_input_latents` was not produced.
- `audio_condition_present`: false.
- `using_audio_conditioning`: false.
- `loss_audio`: 0.0.
- `audio_loss_mask_ratio`: 0.0.
- Cause: data processing emitted warnings that the four smoke MP4 files could not load audio, so native cache had no audio target or audio condition keys. This is a blocker for verifying the real audio branch with this particular smoke metadata. The code path is implemented, but full audio proof still requires a metadata/sample set whose `input_audio` can be loaded.

### G. CFG fused conclusion

- Default smoke keeps `CFG_FUSED=0`.
- `CFG_FUSED=1` was tested with train-only low-res cache and passed.
- Conditional context comes from `inputs_posi`.
- Unconditional/negative context comes from `inputs_nega`.
- If `cfg_fused=True` and `inputs_nega.video_context` or `inputs_nega.audio_context` is missing, the adapter raises a clear `Missing required AnyFlow native key ... in inputs_nega` error including available keys.

### H. Mask conclusion

- Video mask source: `inputs_shared.denoise_mask_video`.
- Audio mask source: `inputs_shared.denoise_mask_audio` when audio target exists.
- Shape check is implemented in `_prepare_loss_mask`: mask rank must match output rank, and each dimension must be either `1` or the output dimension. Wrong rank/dimension now raises with mask/output shapes instead of silently broadcasting.
- `video_loss_mask_ratio`: 0.5 in the low-res/train-only smoke.
- `audio_loss_mask_ratio`: 0.0 because audio target is absent in the current smoke data.

### I. Actual runs

`python -m py_compile examples/ltx2_anyflow_stage1/*.py`

- Result: passed.

Full smoke:

```bash
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-r6-full-codex MAX_STEPS=1 SAVE_STEPS=1 TRAIN_DATASET_REPEAT=1 bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh
```

- Result: data_process completed; 4GPU train failed during `accelerator.backward(loss)`.
- Error summary: `torch.utils.checkpoint.CheckpointError: Recomputed values ... saved metadata shape torch.Size([4096]) ... recomputed metadata shape torch.Size([0])` at positions 25, 30, 79, 84 on all ranks. This is the full-resolution gradient-checkpointing metadata mismatch path.

Low-res smoke:

```bash
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-r6-lowres-codex LOW_RES_SMOKE=1 MAX_STEPS=1 SAVE_STEPS=1 TRAIN_DATASET_REPEAT=1 bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh
```

- Result: passed; generated `native_key_report_step_000001.json`, `anyflow_stage1_log_step_000001.json`, `gradient_sanity_step_000001.json`, and `checkpoint-step_000001/anyflow_wrapper.pt`.

Train-only smoke:

```bash
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-r6-trainonly-codex TRAIN_ONLY=1 CACHE_DIR=./models/train/LTX2.3-I2AV-anyflow-stage1-r6-lowres-codex-cache LOW_RES_SMOKE=1 MAX_STEPS=1 SAVE_STEPS=1 TRAIN_DATASET_REPEAT=1 bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh
```

- Result: passed from the existing native cache.

CFG-fused train-only smoke:

```bash
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-r6-cfg-codex TRAIN_ONLY=1 CACHE_DIR=./models/train/LTX2.3-I2AV-anyflow-stage1-r6-lowres-codex-cache LOW_RES_SMOKE=1 MAX_STEPS=1 SAVE_STEPS=1 TRAIN_DATASET_REPEAT=1 CFG_FUSED=1 CFG_SCALE=1.0 bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh
```

- Result: passed; `using_negative_context=true` in the native key report and stage1 log.

### J. JSON summaries

`native_key_report_step_000001.json` from low-res smoke:

- `video_target_key`: `inputs_shared.input_latents`
- `audio_target_key`: null
- `video_condition_keys_used_in_forward`: `["input_latents_video", "denoise_mask_video"]`
- `audio_condition_keys_used_in_forward`: `[]`
- `ref_condition_keys_used_in_forward`: `[]`
- `using_video_conditioning`: true
- `using_audio_conditioning`: false
- `using_ref_frame_conditioning`: false
- `using_negative_context`: false
- `audio_present`: false
- `audio_target_present`: false
- `audio_condition_present`: false
- `audio_fallback_reason`: `audio_input_latents missing from native cache/unit outputs`

`anyflow_stage1_log_step_000001.json` from low-res smoke:

- `loss_total`: 1.4002866744995117
- `loss_video`: 1.4002866744995117
- `loss_audio`: 0.0
- `audio_present`: false
- `audio_loss_weight`: 1.0
- `video_loss_mask_ratio`: 0.5
- `audio_loss_mask_ratio`: 0.0
- `using_video_loss_mask`: 1.0
- `using_audio_loss_mask`: 0.0
- `using_video_conditioning`: true
- `using_audio_conditioning`: false
- `using_negative_context`: false
- `u_norm_video`: 90.06635284423828
- `target_norm_video`: 121.17963409423828
- `u_norm_audio`: 0.0
- `target_norm_audio`: 0.0

`gradient_sanity_step_000001.json` from low-res smoke:

- `total_trainable_tensors`: 2440
- `trainable_with_grad_count`: 2440
- `trainable_without_grad_count`: 0
- `trainable_with_nonzero_grad_count`: 394
- `trainable_zero_grad_count`: 2046
- `trainable_without_grad_names` first 20: `[]`
- `trainable_zero_grad_names` first 20: `["pipe.dit.dit.audio_adaln_single.gate", "pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", "pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", "pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", "pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", "pipe.dit.dit.audio_prompt_adaln_single.gate", "pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", "pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", "pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", "pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", "pipe.dit.dit.av_ca_video_scale_shift_adaln_single.gate", "pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", "pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", "pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", "pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", "pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.gate", "pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", "pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", "pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", "pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias"]`
- `grad_norm_top20` first 20: `[["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", 4.424962997436523], ["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", 3.9781365394592285], ["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", 1.6359113454818726], ["pipe.dit.dit.adaln_single.gate", 0.6015625], ["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", 0.3508561849594116], ["pipe.dit.dit.prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", 0.12856413424015045], ["pipe.dit.dit.prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", 0.09583821147680283], ["pipe.dit.dit.transformer_blocks.22.attn1.to_v.lora_B.weight", 0.031188003718852997], ["pipe.dit.dit.transformer_blocks.21.attn1.to_v.lora_B.weight", 0.027973035350441933], ["pipe.dit.dit.transformer_blocks.22.attn1.to_out.0.lora_B.weight", 0.02425260841846466], ["pipe.dit.dit.transformer_blocks.19.attn1.to_v.lora_B.weight", 0.023984255269169807], ["pipe.dit.dit.transformer_blocks.17.attn1.to_out.0.lora_B.weight", 0.022808268666267395], ["pipe.dit.dit.transformer_blocks.20.attn1.to_v.lora_B.weight", 0.021834535524249077], ["pipe.dit.dit.transformer_blocks.0.attn1.to_v.lora_B.weight", 0.021269800141453743], ["pipe.dit.dit.transformer_blocks.18.attn1.to_out.0.lora_B.weight", 0.021209418773651123], ["pipe.dit.dit.transformer_blocks.21.attn1.to_out.0.lora_B.weight", 0.019816014915704727], ["pipe.dit.dit.transformer_blocks.0.attn1.to_out.0.lora_B.weight", 0.019594257697463036], ["pipe.dit.dit.prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", 0.019576257094740868], ["pipe.dit.dit.transformer_blocks.23.attn1.to_v.lora_B.weight", 0.018594402819871902], ["pipe.dit.dit.transformer_blocks.19.attn1.to_out.0.lora_B.weight", 0.018376661464571953]]`

CFG-fused key/log summary:

- `using_negative_context`: true
- `loss_total`: 15.6661376953125
- `video_condition_keys_used_in_forward`: `["input_latents_video", "denoise_mask_video"]`
- `audio_present`: false

### K. Remaining limitations

- Stage 1 only, no DMD/on-policy.
- Full AnyFlow reproduction still requires Stage 2 flow-map backward simulation + DMD.
- Full-resolution smoke currently fails on the gradient-checkpointing recompute metadata mismatch described above; current workaround is `LOW_RES_SMOKE=1` or disabling the problematic full-resolution gradient checkpointing path.
- Audio branch is implemented but not truly verified by this smoke metadata because all four smoke MP4 files failed audio loading and no `audio_input_latents`/audio condition keys were produced. This is a blocker for claiming real audio branch participation; use a smoke metadata set with loadable audio to verify nonzero `loss_audio`.

## Round 7: Native LTX2 data alignment and low-res smoke verification

### A. Files changed

- `examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-lyh_4gpu.sh`
- `examples/ltx2_anyflow_stage1/codex_work.md`

### B. Files outside this directory

No files outside examples/ltx2_anyflow_stage1 were modified.

### C. Reference script alignment

- Reference script: `examples/ltx2/model_training/full/LTX-2.3-I2AV-splited_lyh_4gpu.sh`.
- The new AnyFlow smoke script keeps the same native I2AV structure:
  - `data_file_keys="video,input_audio"`
  - `extra_inputs="input_audio,input_image"`
  - 4-GPU ZeRO3 offload via `examples/ltx2/model_training/full/accelerate_config_zero3offload_4gpu.yaml`
  - LoRA on `dit` with target modules `to_k,to_q,to_v,to_out.0`
- Low-res smoke uses `metadata_lyh_smoke4.csv` and explicitly sets `LOW_RES_SMOKE=1`.

### D. Native key report summary

`native_key_report_step_000001.json` from the low-res smoke:

- `using_video_conditioning`: true
- `using_audio_conditioning`: false
- `audio_present`: false
- `audio_target_present`: false
- `audio_condition_present`: false
- `using_negative_context`: false
- `video_condition_keys_used_in_forward`: `["input_latents_video", "denoise_mask_video"]`
- `audio_condition_keys_used_in_forward`: `[]`
- `optional_condition_keys_found`: `["input_latents_video", "denoise_mask_video"]`
- `optional_condition_keys_missing`: `["ref_frames_latents", "ref_frames_positions", "in_context_video_latents", "in_context_video_positions", "input_latents_audio", "denoise_mask_audio"]`
- `audio_fallback_reason`: `audio_input_latents missing from native cache/unit outputs`

### E. AnyFlow stage1 log summary

`anyflow_stage1_log_step_000001.json` from the low-res smoke:

- `loss_total`: 3.8371195793151855
- `loss_video`: 3.8371195793151855
- `loss_audio`: 0.0
- `audio_present`: false
- `audio_condition_present`: false
- `audio_fallback_reason`: `audio_input_latents missing from native cache/unit outputs`
- `video_loss_mask_ratio`: 0.5
- `audio_loss_mask_ratio`: 0.0
- `using_video_loss_mask`: 1.0
- `using_audio_loss_mask`: 0.0
- `using_video_conditioning`: true
- `using_audio_conditioning`: false
- `using_negative_context`: false

### F. Gradient sanity summary

`gradient_sanity_step_000001.json` from the low-res smoke:

- `total_trainable_tensors`: 2440
- `trainable_with_grad_count`: 2440
- `trainable_without_grad_count`: 0
- `trainable_with_nonzero_grad_count`: 394
- `trainable_zero_grad_count`: 2046
- `trainable_without_grad_names` first 20: `[]`
- `trainable_zero_grad_names` first 20: `["pipe.dit.dit.audio_adaln_single.gate", "pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", "pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", "pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", "pipe.dit.dit.audio_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", "pipe.dit.dit.audio_prompt_adaln_single.gate", "pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", "pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", "pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", "pipe.dit.dit.audio_prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", "pipe.dit.dit.av_ca_video_scale_shift_adaln_single.gate", "pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", "pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", "pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", "pipe.dit.dit.av_ca_video_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", "pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.gate", "pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", "pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", "pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", "pipe.dit.dit.av_ca_audio_scale_shift_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias"]`
- `grad_norm_top20` first 20: `[["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", 3.0488951206207275], ["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", 1.8966784477233887], ["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", 1.4212859869003296], ["pipe.dit.dit.adaln_single.gate", 0.87109375], ["pipe.dit.dit.adaln_single.r_adaln.emb.timestep_embedder.linear_1.bias", 0.2652214765548706], ["pipe.dit.dit.prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_1.weight", 0.13142144680023193], ["pipe.dit.dit.prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.weight", 0.0798197016119957], ["pipe.dit.dit.transformer_blocks.25.attn1.to_out.0.lora_B.weight", 0.02393363043665886], ["pipe.dit.dit.transformer_blocks.45.attn2.to_q.lora_B.weight", 0.02351904846727848], ["pipe.dit.dit.transformer_blocks.36.attn2.to_q.lora_B.weight", 0.02220844477415085], ["pipe.dit.dit.transformer_blocks.24.attn1.to_out.0.lora_B.weight", 0.019869552925229073], ["pipe.dit.dit.prompt_adaln_single.r_adaln.emb.timestep_embedder.linear_2.bias", 0.019526707008481026], ["pipe.dit.dit.transformer_blocks.0.attn1.to_v.lora_B.weight", 0.01924874261021614], ["pipe.dit.dit.transformer_blocks.25.attn1.to_v.lora_B.weight", 0.019079284742474556], ["pipe.dit.dit.transformer_blocks.23.attn1.to_out.0.lora_B.weight", 0.016496745869517326], ["pipe.dit.dit.transformer_blocks.22.attn1.to_out.0.lora_B.weight", 0.01582866534590721], ["pipe.dit.dit.transformer_blocks.43.attn2.to_q.lora_B.weight", 0.015129436738789082], ["pipe.dit.dit.transformer_blocks.35.attn2.to_q.lora_B.weight", 0.01489232387393713], ["pipe.dit.dit.transformer_blocks.24.attn1.to_v.lora_B.weight", 0.014796508476138115], ["pipe.dit.dit.transformer_blocks.23.attn1.to_v.lora_B.weight", 0.014625592157244682]`

### G. Actual low-res smoke command and result

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_NAME=LTX2.3-I2AV-anyflow-stage1-lora-lowres-smoke \
LOW_RES_SMOKE=1 \
HEIGHT=128 \
WIDTH=128 \
NUM_FRAMES=9 \
MAX_STEPS=1 \
SAVE_STEPS=1 \
TRAIN_DATASET_REPEAT=4 \
bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-lyh_4gpu.sh
```

Result: passed. Generated `native_key_report_step_000001.json`, `anyflow_stage1_log_step_000001.json`, `gradient_sanity_step_000001.json`, and `checkpoint-step_000001/anyflow_wrapper.pt`.

### H. Full-res / TRAIN_ONLY

- Full-res execution of the new script still hits the same PyTorch checkpoint recompute metadata mismatch seen earlier.
- `TRAIN_ONLY=1` remains supported by the smoke script, but the current verification for this round was the explicit low-res run above.

### I. Remaining limitations

- Stage 1 only, no DMD/on-policy.
- Full AnyFlow reproduction still requires Stage 2 flow-map backward simulation plus DMD.
- Audio branch is still not truly exercised by this smoke metadata because the native audio operator could not load audio from the referenced MP4 files, so `audio_present=false`.

## Round 8: Native LoRA checkpoint sampling compatibility (2026-06-11)

A. Modified files:
- `examples/ltx2_anyflow_stage1/sample_ltx2_anyflow_stage1.py`
- `examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py`
- `examples/ltx2_anyflow_stage1/README.md`
- `examples/ltx2_anyflow_stage1/codex_work.md`

B. Files outside `examples/ltx2_anyflow_stage1`:
No files outside `examples/ltx2_anyflow_stage1` were modified.

C. Native sidecar checkpoint sampling fix:
- The sampler now reads `anyflow_wrapper.pt` once before wrapper construction.
- LoRA is enabled if `use_lora=true`, or `lora_base_model="dit"`, or the checkpoint state dict contains `lora_A`/`lora_B` keys.
- Native `lora_target_modules` takes priority over legacy `lora_target_filter`; if neither exists, the sampler defaults to `to_k,to_q,to_v,to_out.0`.
- `lora_alpha` falls back to `lora_scale`, then to `lora_rank`.
- Non-DiT `lora_base_model` values now raise a clear `RuntimeError` because this Stage 1 wrapper only supports DiT LoRA.
- If LoRA is required but injection matches zero Linear layers, sampling raises instead of silently continuing.

D. Native training LoRA default:
- `anyflow_stage1` and `anyflow_stage1:train` now default to effective `lora_base_model="dit"` when the user does not pass a LoRA base model.
- `anyflow_stage1:data_process` keeps effective `lora_base_model=None` and does not install LoRA.
- Native parser defaults are LoRA-friendly: `lora_rank=256`, `lora_target_modules=to_k,to_q,to_v,to_out.0`; `lora_alpha` still falls back to rank.
- Existing smoke scripts can continue overriding rank/alpha to 8.

E. `anyflow_config.json` compatibility aliases:
- Native checkpoints still save `lora_base_model`, `lora_target_modules`, `lora_rank`, and `lora_alpha`.
- Native checkpoints now also save legacy aliases `use_lora` and `lora_target_filter`.

F. README update:
- README now states native train tasks default to DiT LoRA while data_process does not need LoRA.
- README now describes native/legacy/state-dict LoRA detection for sampling.
- README command examples are written for the current 4-GPU environment; 8-GPU configuration is not assumed in current smoke/formal examples.

G. Validation commands run:
- Passed: `python -m py_compile examples/ltx2_anyflow_stage1/*.py`
- Passed: pure-Python `resolve_lora_settings` checks for native config, legacy config, state-dict inference, native-over-legacy target priority, and `lora_alpha` fallback to rank.
- Pure-Python check output summary: native enabled from `lora_base_model=dit`; legacy enabled from `use_lora=True`; state-dict enabled from `lora_A`; native `lora_target_modules` wins over legacy `lora_target_filter`; missing alpha falls back to rank.

H. GPU/model-weight smoke:
- Not run in this round. The requested minimum validation is compile plus pure-Python LoRA config parsing; no fresh real model checkpoint path was provided for latent-only sampling.

I. Remaining limitations:
- Stage 1 only: no DMD, no discriminator, no real_score/fake_score, no on-policy distillation.
- Full native checkpoint sampling still requires a valid LTX2 model config/weights and a real native sidecar checkpoint.
