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
