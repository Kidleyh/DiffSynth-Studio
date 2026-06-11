#!/usr/bin/env bash
set -Eeuo pipefail

# Probe whether full-resolution AnyFlow Stage 1 fails because LTX2 gradient
# checkpointing is enabled, or because the no-checkpoint path simply OOMs.
#
# This intentionally keeps the native I2AV data/cache/model path and ZeRO3
# config, but forces USE_GRADIENT_CHECKPOINTING=0 at full resolution.

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export RUN_NAME=${RUN_NAME:-LTX2.3-I2AV-anyflow-stage1-lora-fullres-nogc-oom-probe}
export LOW_RES_SMOKE=0
export HEIGHT=${HEIGHT:-512}
export WIDTH=${WIDTH:-768}
export NUM_FRAMES=${NUM_FRAMES:-121}
export FRAME_RATE=${FRAME_RATE:-25}
export USE_GRADIENT_CHECKPOINTING=0
export MAX_STEPS=${MAX_STEPS:-1}
export SAVE_STEPS=${SAVE_STEPS:-1}
export TRAIN_DATASET_REPEAT=${TRAIN_DATASET_REPEAT:-1}
export DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-0}

echo "Full-resolution no-gradient-checkpointing OOM probe"
echo "RUN_NAME=$RUN_NAME"
echo "HEIGHT=$HEIGHT WIDTH=$WIDTH NUM_FRAMES=$NUM_FRAMES FRAME_RATE=$FRAME_RATE"
echo "USE_GRADIENT_CHECKPOINTING=$USE_GRADIENT_CHECKPOINTING"
echo "TRAIN_DATASET_REPEAT=$TRAIN_DATASET_REPEAT MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS"

exec bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-lyh_4gpu.sh
