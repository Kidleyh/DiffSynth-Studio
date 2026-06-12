#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export USE_GRADIENT_CHECKPOINTING=1

RUN_NAME=${RUN_NAME:-LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-4gpu-zero2}
METADATA=${METADATA:-examples/ltx2/model_training/full/metadata_lyh_smoke4.csv}
OUTPUT_DIR=${OUTPUT_DIR:-./models/train/${RUN_NAME}}
CACHE_DIR=${CACHE_DIR:-./models/train/${RUN_NAME}-cache}
LOG_DIR=${LOG_DIR:-./models/train/${RUN_NAME}-logs}
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_DIR"

env | sort > "$LOG_DIR/env.txt"
: > "$LOG_DIR/command.txt"

HEIGHT=${HEIGHT:-128}
WIDTH=${WIDTH:-128}
NUM_FRAMES=${NUM_FRAMES:-9}
FRAME_RATE=${FRAME_RATE:-25}
DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-0}
TRAIN_DATASET_REPEAT=${TRAIN_DATASET_REPEAT:-1}
MAX_STEPS=${MAX_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-8}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-to_k,to_q,to_v,to_out.0}
TRAIN_ONLY=${TRAIN_ONLY:-0}

ENCODER_MODEL_SPEC=${ENCODER_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors}
TRANSFORMER_MODEL_SPEC=${TRANSFORMER_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-examples/ltx2/model_training/full/accelerate_config_zero2offload_4gpu.yaml}

COMMON_ARGS=(
  --data_file_keys "video,input_audio"
  --extra_inputs "input_audio,input_image"
  --height "$HEIGHT"
  --width "$WIDTH"
  --num_frames "$NUM_FRAMES"
  --frame_rate "$FRAME_RATE"
  --dataset_num_workers "$DATASET_NUM_WORKERS"
  --trainable_models "dit"
  --remove_prefix_in_ckpt "pipe.dit."
  --use_gradient_checkpointing
)

write_command() {
  local label=$1
  shift
  {
    printf '[%s]\n' "$label"
    printf '%q ' "$@"
    printf '\n\n'
  } >> "$LOG_DIR/command.txt"
}

python -m py_compile examples/ltx2_anyflow_stage1/*.py

echo "[CONFIG] 4GPU ZeRO2/offload GC diagnostic"
echo "[CONFIG] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[CONFIG] ACCELERATE_CONFIG=$ACCELERATE_CONFIG"
echo "[CONFIG] RUN_NAME=$RUN_NAME"
echo "[CONFIG] OUTPUT_DIR=$OUTPUT_DIR"
echo "[CONFIG] CACHE_DIR=$CACHE_DIR"
echo "[CONFIG] LOG_DIR=$LOG_DIR"
echo "[CONFIG] HEIGHT=$HEIGHT WIDTH=$WIDTH NUM_FRAMES=$NUM_FRAMES MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS"

if [[ "$TRAIN_ONLY" == "1" ]]; then
  echo "[1/2] TRAIN_ONLY=1, reusing native cache at $CACHE_DIR"
else
  DATA_PROCESS_CMD=(
    accelerate launch --num_processes 1 examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py
    --dataset_base_path ""
    --dataset_metadata_path "$METADATA"
    "${COMMON_ARGS[@]}"
    --dataset_repeat 1
    --model_id_with_origin_paths "$ENCODER_MODEL_SPEC"
    --learning_rate "$LEARNING_RATE"
    --weight_decay "$WEIGHT_DECAY"
    --num_epochs 1
    --output_path "$CACHE_DIR"
    --task "anyflow_stage1:data_process"
  )
  write_command data_process "${DATA_PROCESS_CMD[@]}"
  echo "[1/2] AnyFlow native data_process -> $CACHE_DIR"
  "${DATA_PROCESS_CMD[@]}" 2>&1 | tee "$LOG_DIR/data_process.log"
fi

TRAIN_CMD=(
  accelerate launch --config_file "$ACCELERATE_CONFIG" examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py
  --dataset_base_path "$CACHE_DIR"
  "${COMMON_ARGS[@]}"
  --dataset_repeat "$TRAIN_DATASET_REPEAT"
  --model_id_with_origin_paths "$TRANSFORMER_MODEL_SPEC"
  --initialize_model_on_cpu
  --learning_rate "$LEARNING_RATE"
  --weight_decay "$WEIGHT_DECAY"
  --num_epochs 1
  --save_steps "$SAVE_STEPS"
  --max_steps "$MAX_STEPS"
  --output_path "$OUTPUT_DIR"
  --lora_base_model "dit"
  --lora_target_modules "$LORA_TARGET_MODULES"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --save_gradient_sanity
  --task "anyflow_stage1:train"
)
write_command train "${TRAIN_CMD[@]}"
echo "[2/2] AnyFlow native 4GPU ZeRO2/offload GC train -> $OUTPUT_DIR"
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_DIR/train.log"

CKPT="${OUTPUT_DIR}/checkpoint-step_000001"
LOG_JSON="${OUTPUT_DIR}/anyflow_stage1_log_step_000001.json"
test -f "${CKPT}/anyflow_wrapper.pt"
test -f "${CKPT}/anyflow_config.json"
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

echo "[OK] Low-res 4GPU ZeRO2/offload gradient-checkpointing train-1-step test passed."
