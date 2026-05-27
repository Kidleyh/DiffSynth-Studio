# AnyFlow-LTX2 Stage 1

This directory implements only AnyFlow Stage 1: Forward Flow Map Training for LTX/LTX2. It does not implement DMD, real_score, fake_score, discriminator training, or on-policy distillation.

The goal is to add an `r_timestep` condition around the existing LTX2 DiT without modifying DiffSynth source files, then train a direct flow-map transition:

`z_r = z_t - (t - r) * u_theta(z_t, t, r)`

## Files

- `anyflow_ltx2_scheduler.py`: normalized continuous-time Euler scheduler for training noise injection and any-step sampling.
- `anyflow_ltx2_model_wrapper.py`: wraps the original `LTXModel`, copies its timestep embedding path for `r_timestep`, and blends `gate * emb(t) + (1 - gate) * emb_r(r)`.
- `anyflow_ltx2_stage1_loss.py`: Stage-1 central-difference target and video/audio loss.
- `train_ltx2_anyflow_stage1.py`: minimal latent-cache trainer with smoke-test mode.
- `sample_ltx2_anyflow_stage1.py`: minimal sampler using the new flow-map Euler scheduler.

## Data Format

The first trainer version expects latent-cache metadata as CSV or JSONL. Each row should contain:

- `video_latents`: path to a `.pt` video latent tensor.
- `audio_latents`: path to a `.pt` audio latent tensor.
- `video_positions`: path to LTX2 video positions tensor.
- `audio_positions`: path to LTX2 audio positions tensor.
- `prompt` or cached context tensors.
- `prompt_embeds`, or separate `video_context` and `audio_context`.

Relative paths are resolved under `--latent_cache_dir`.

## Smoke Test

```bash
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio
python examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1.py \
  --output_dir examples/ltx2_anyflow_stage1/smoke_out \
  --smoke_test \
  --batch_size 1 \
  --max_steps 2 \
  --save_steps 2 \
  --dtype fp32
```

## Training

```bash
python examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1.py \
  --model_config_path examples/ltx2_anyflow_stage1/model_configs_ltx23.json \
  --dataset_metadata /path/to/metadata.csv \
  --latent_cache_dir /path/to/latent_cache \
  --output_dir /path/to/anyflow_stage1_out \
  --height 512 --width 768 --num_frames 121 --frame_rate 24 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --learning_rate 1e-5 \
  --max_steps 10000 \
  --save_steps 500 \
  --dtype bf16 \
  --audio_loss_weight 1.0 \
  --boundary_prob 0.5 \
  --fd_eps 0.001 \
  --cfg_fused \
  --cfg_scale 1.0
```

## Sampling

```bash
for steps in 4 8 16 32; do
  python examples/ltx2_anyflow_stage1/sample_ltx2_anyflow_stage1.py \
    --model_config_path examples/ltx2_anyflow_stage1/model_configs_ltx23.json \
    --checkpoint /path/to/anyflow_stage1_out/checkpoint-step_10000 \
    --prompt "A beautiful sunset over the ocean." \
    --output_path anyflow_stage1_${steps}step.mp4 \
    --num_inference_steps ${steps} \
    --cfg_scale 1.0 \
    --height 512 --width 768 --num_frames 121 --frame_rate 24 \
    --dtype bf16
done
```

## Known Limits

Stage 1 provides any-step flow-map initialization for LTX2. It cannot reproduce final AnyFlow metrics by itself because Stage 2 on-policy DMD/distillation is intentionally not included here.

