#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

RUN_NAME=${RUN_NAME:-LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-test}
OUTPUT_DIR=${OUTPUT_DIR:-./models/train/${RUN_NAME}}
CACHE_DIR=${CACHE_DIR:-./models/train/${RUN_NAME}-cache}
LOG_DIR=${LOG_DIR:-./models/train/${RUN_NAME}-logs}

python -m py_compile examples/ltx2_anyflow_stage1/*.py

LOW_RES_SMOKE=1 \
USE_GRADIENT_CHECKPOINTING=1 \
HEIGHT=${HEIGHT:-128} \
WIDTH=${WIDTH:-128} \
NUM_FRAMES=${NUM_FRAMES:-9} \
MAX_STEPS=${MAX_STEPS:-1} \
SAVE_STEPS=${SAVE_STEPS:-1} \
TRAIN_DATASET_REPEAT=${TRAIN_DATASET_REPEAT:-1} \
RUN_NAME="$RUN_NAME" \
CACHE_DIR="$CACHE_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
LOG_DIR="$LOG_DIR" \
bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh

CKPT="${OUTPUT_DIR}/checkpoint-step_000001"
LOG_JSON="${OUTPUT_DIR}/anyflow_stage1_log_step_000001.json"
test -f "${CKPT}/anyflow_wrapper.pt"
test -f "${CKPT}/anyflow_config.json"
test -f "${OUTPUT_DIR}/gradient_sanity_step_000001.json"
test -f "${OUTPUT_DIR}/native_key_report_step_000001.json"
test -f "$LOG_JSON"

python - <<PY
import json
import math
from pathlib import Path
log = json.loads(Path("${LOG_JSON}").read_text())
loss_total = float(log["loss_total"])
if not math.isfinite(loss_total):
    raise SystemExit(f"loss_total is not finite: {loss_total!r}")
main = float(log.get("gradient_checkpointing_main_forward", -999))
target = float(log.get("gradient_checkpointing_target_forward", -999))
uncond = float(log.get("gradient_checkpointing_uncond_forward", -999))
if main != 1.0:
    raise SystemExit(f"expected gradient_checkpointing_main_forward=1.0, got {main}")
if target != 0.0:
    raise SystemExit(f"expected gradient_checkpointing_target_forward=0.0, got {target}")
if uncond not in (-1.0, 0.0):
    raise SystemExit(f"expected gradient_checkpointing_uncond_forward=-1.0 or 0.0, got {uncond}")
print(f"[CHECK] loss_total finite: {loss_total}")
print(f"[CHECK] checkpoint policy: main={main} target={target} uncond={uncond}")
PY

echo "[OK] Low-res 4-GPU gradient-checkpointing train-1-step test passed."
