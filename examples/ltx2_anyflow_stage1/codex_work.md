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

