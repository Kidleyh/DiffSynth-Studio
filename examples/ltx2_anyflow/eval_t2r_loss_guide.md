# AnyFlow t->r Loss Evaluation

This note explains what `eval_t2r_loss.py` measures and how to read its output.

## Goal

`eval_t2r_loss.py` checks whether the model has learned the AnyFlow-style one-step mapping from a noisy endpoint `t` to a cleaner endpoint `r`.

It does not judge final video quality directly. Instead, it evaluates the latent-space objective that AnyFlow is trying to improve.

## Evaluation Setup

The script compares two checkpoints:

- `base`: the original LTX2 transformer checkpoint.
- `trained`: the AnyFlow-trained checkpoint specified by `TRAINED_CKPT`.

It reads validation samples from cached features. The helper script builds this cache automatically if needed:

```bash
examples/ltx2_anyflow/eval/cache_val
```

The default validation metadata is:

```bash
data/val.csv
```

## What Happens For Each Sample

For each validation sample, the script starts from the clean latent:

```text
x_0 = original latent
epsilon = random noise
```

It then constructs two points on the same linear noise path:

```text
x_t = t * epsilon + (1 - t) * x_0
x_r = r * epsilon + (1 - r) * x_0
```

The model receives `x_t`, `t`, and `r`, then predicts a velocity. The script applies one AnyFlow step:

```text
pred_x_r = x_t - (t - r) * pred_velocity
```

Finally, it compares:

```text
pred_x_r vs true x_r
```

This directly measures whether the model can move from `t` to `r` in one step.

## Output File

The default output CSV is:

```bash
examples/ltx2_anyflow/eval/t2r_loss.csv
```

Each row contains one checkpoint, one sample, and one `(t, r)` pair.

Fields:

```text
checkpoint
sample_id
t
r
video_velocity_mse
video_reconstruction_mse
audio_velocity_mse
audio_reconstruction_mse
```

## Metrics

`video_velocity_mse`

MSE between the predicted video velocity and the theoretical target velocity:

```text
epsilon - x_0
```

This is close to the usual diffusion or flow-matching training target.

`video_reconstruction_mse`

MSE between the predicted one-step result and the true `x_r`:

```text
pred_x_r vs true x_r
```

This is the most important AnyFlow metric.

`audio_velocity_mse`

The same velocity MSE, but for audio latents.

`audio_reconstruction_mse`

The same t-to-r reconstruction MSE, but for audio latents.

## Default t,r Pairs

The default pairs are:

```text
0.8 -> 0.8
0.8 -> 0.4
0.8 -> 0.2
0.8 -> 0.0
0.5 -> 0.0
```

Interpretation:

- `0.8 -> 0.8`: sanity check; almost no movement.
- `0.8 -> 0.4`: medium jump.
- `0.8 -> 0.2`: large jump.
- `0.8 -> 0.0`: high-noise to clean latent in one step.
- `0.5 -> 0.0`: medium-noise to clean latent in one step.

The cross-step pairs are the important ones for AnyFlow.

## Expected Signal

For the same `(t, r)` pair, compare:

```text
base vs trained
```

The desired result is:

```text
trained video_reconstruction_mse < base video_reconstruction_mse
trained audio_reconstruction_mse < base audio_reconstruction_mse
```

The most important pairs are:

```text
0.8 -> 0.4
0.8 -> 0.2
0.8 -> 0.0
```

If the trained checkpoint improves on these pairs, it suggests the model is learning the AnyFlow `t -> r` behavior.

## Run Command

Example:

```bash
cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio

TRAINED_CKPT=examples/ltx2_anyflow/output/ltx2_anyflow_0430/step-50000.safetensors \
bash examples/ltx2_anyflow/eval/run_eval_t2r_loss.sh
```

By default, the script evaluates 16 validation samples. To evaluate more:

```bash
MAX_SAMPLES=64 \
TRAINED_CKPT=examples/ltx2_anyflow/output/ltx2_anyflow_0430/step-50000.safetensors \
bash examples/ltx2_anyflow/eval/run_eval_t2r_loss.sh
```

## Summarize Results

Use this command to print mean metrics by checkpoint and `(t, r)`:

```bash
python - <<'PY'
import pandas as pd

p = "examples/ltx2_anyflow/eval/t2r_loss.csv"
df = pd.read_csv(p)

print(df.groupby(["checkpoint", "t", "r"])[[
    "video_velocity_mse",
    "video_reconstruction_mse",
    "audio_velocity_mse",
    "audio_reconstruction_mse",
]].mean())
PY
```

Focus first on `video_reconstruction_mse` and `audio_reconstruction_mse`.
