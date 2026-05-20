#!/usr/bin/env bash
set -Eeuo pipefail

# One-sample AnyFlow/LTX2 smoke test.
# This version uses zhangyan's existing transformer checkpoint for stage 2.
# If you do NOT want to use zhangyan's checkpoint, either:
#   1) run run_smoke_default_download.sh instead, or
#   2) delete the TRANSFORMER_MODEL_SPEC=... prefix below so run_one_sample.sh uses its default download spec.
# Stage 1 encoder/tokenizer weights may still download if they are not cached locally.

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio
source /root/miniconda3/etc/profile.d/conda.sh
conda activate far

TRANSFORMER_MODEL_SPEC=/gemini/platform/public/aigc/human_guozz2/code/zhangyan/DiffSynth-Studio-LTX/model_output/LTX2.3-I2AV-moe-0519/step-3000.safetensors \
bash examples/ltx2_anyflow/smoke/run_one_sample.sh
