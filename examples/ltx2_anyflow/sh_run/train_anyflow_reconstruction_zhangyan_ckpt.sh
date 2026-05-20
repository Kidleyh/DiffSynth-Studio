#!/usr/bin/env bash
set -Eeuo pipefail

# Same full reconstruction training script, but defaults stage-2 transformer to zhangyan's checkpoint.
# You can override any variable before running, e.g.:
#   NUM_EPOCHS=2 SAVE_STEPS=500 ANYFLOW_RECONSTRUCTION_WEIGHT=0.05 bash ...

export TRANSFORMER_MODEL_SPEC=${TRANSFORMER_MODEL_SPEC:-/gemini/platform/public/aigc/human_guozz2/code/zhangyan/DiffSynth-Studio-LTX/model_output/LTX2.3-I2AV-moe-0519/step-3000.safetensors}
exec bash /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio/examples/ltx2_anyflow/sh_run/train_anyflow_reconstruction_full.sh
