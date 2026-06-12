#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export USE_GRADIENT_CHECKPOINTING=1

RUN_NAME=${RUN_NAME:-LTX2.3-I2AV-anyflow-stage1-lowres-gc-train1-1gpu-nodeepspeed}
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

echo "[CONFIG] 1GPU no-DeepSpeed GC diagnostic"
echo "[CONFIG] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
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
  accelerate launch --num_processes 1 examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py
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
  --allow_trainable_without_grad
  --task "anyflow_stage1:train"
)
write_command train "${TRAIN_CMD[@]}"
echo "[2/2] AnyFlow native 1GPU no-DeepSpeed GC train -> $OUTPUT_DIR"
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_DIR/train.log"

echo "[OK] Low-res 1GPU no-DeepSpeed gradient-checkpointing train-1-step test passed."
