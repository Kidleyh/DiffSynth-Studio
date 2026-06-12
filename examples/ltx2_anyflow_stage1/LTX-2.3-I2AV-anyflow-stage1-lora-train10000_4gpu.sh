#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN_NAME=${RUN_NAME:-LTX2.3-I2AV-anyflow-stage1-lora-train10000-4gpu}
METADATA=${METADATA:-examples/ltx2/model_training/full/metadata_lyh_smoke4.csv}
CACHE_DIR=${CACHE_DIR:-./models/train/${RUN_NAME}-cache}
OUTPUT_DIR=${OUTPUT_DIR:-./models/train/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-./models/train/${RUN_NAME}-logs}
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# Full-resolution defaults aligned with the native LTX2 I2AV 4-GPU path.
LOW_RES_SMOKE=${LOW_RES_SMOKE:-0}
TRAIN_ONLY=${TRAIN_ONLY:-0}
CFG_FUSED=${CFG_FUSED:-0}
CFG_SCALE=${CFG_SCALE:-1.0}
if [[ "$LOW_RES_SMOKE" == "1" ]]; then
  HEIGHT=${HEIGHT:-128}
  WIDTH=${WIDTH:-128}
  NUM_FRAMES=${NUM_FRAMES:-9}
else
  HEIGHT=${HEIGHT:-512}
  WIDTH=${WIDTH:-768}
  NUM_FRAMES=${NUM_FRAMES:-121}
fi
if [[ "$HEIGHT" == "512" && "$WIDTH" == "768" && "$NUM_FRAMES" == "121" ]]; then
  USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-1}
else
  USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-0}
fi

FRAME_RATE=${FRAME_RATE:-25}
DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-0}
TRAIN_DATASET_REPEAT=${TRAIN_DATASET_REPEAT:-25}
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
LORA_RANK=${LORA_RANK:-256}
LORA_ALPHA=${LORA_ALPHA:-$LORA_RANK}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-to_k,to_q,to_v,to_out.0}

ENCODER_MODEL_SPEC=${ENCODER_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors}
TRANSFORMER_MODEL_SPEC=${TRANSFORMER_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-examples/ltx2/model_training/full/accelerate_config_zero3offload_4gpu.yaml}

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
)
if [[ "$USE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  COMMON_ARGS+=(--use_gradient_checkpointing)
fi

printf '[CONFIG] RUN_NAME=%s\n' "$RUN_NAME"
printf '[CONFIG] CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf '[CONFIG] METADATA=%s\n' "$METADATA"
printf '[CONFIG] CACHE_DIR=%s\n' "$CACHE_DIR"
printf '[CONFIG] OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
printf '[CONFIG] HEIGHT=%s WIDTH=%s NUM_FRAMES=%s FRAME_RATE=%s\n' "$HEIGHT" "$WIDTH" "$NUM_FRAMES" "$FRAME_RATE"
printf '[CONFIG] MAX_STEPS=%s SAVE_STEPS=%s TRAIN_DATASET_REPEAT=%s\n' "$MAX_STEPS" "$SAVE_STEPS" "$TRAIN_DATASET_REPEAT"
printf '[CONFIG] LORA_RANK=%s LORA_ALPHA=%s LORA_TARGET_MODULES=%s\n' "$LORA_RANK" "$LORA_ALPHA" "$LORA_TARGET_MODULES"
printf '[CONFIG] USE_GRADIENT_CHECKPOINTING=%s TRAIN_ONLY=%s CFG_FUSED=%s\n' "$USE_GRADIENT_CHECKPOINTING" "$TRAIN_ONLY" "$CFG_FUSED"

python -m py_compile examples/ltx2_anyflow_stage1/*.py

echo "[1/2] AnyFlow native data_process -> $CACHE_DIR"
if [[ "$TRAIN_ONLY" == "1" ]]; then
  echo "[1/2] TRAIN_ONLY=1, reusing native cache at $CACHE_DIR"
else
  accelerate launch --num_processes 1 examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py \
    --dataset_base_path "" \
    --dataset_metadata_path "$METADATA" \
    "${COMMON_ARGS[@]}" \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "$ENCODER_MODEL_SPEC" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --num_epochs 1 \
    --output_path "$CACHE_DIR" \
    --task "anyflow_stage1:data_process" \
    2>&1 | tee "$LOG_DIR/data_process.log"
fi

ANYFLOW_TRAIN_ARGS=(--save_gradient_sanity)
if [[ "$CFG_FUSED" == "1" ]]; then
  ANYFLOW_TRAIN_ARGS+=(--cfg_fused --cfg_scale "$CFG_SCALE")
fi

echo "[2/2] AnyFlow native 4-GPU LoRA train -> $OUTPUT_DIR"
accelerate launch --config_file "$ACCELERATE_CONFIG" examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py \
  --dataset_base_path "$CACHE_DIR" \
  "${COMMON_ARGS[@]}" \
  --dataset_repeat "$TRAIN_DATASET_REPEAT" \
  --model_id_with_origin_paths "$TRANSFORMER_MODEL_SPEC" \
  --initialize_model_on_cpu \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --num_epochs 1 \
  --save_steps "$SAVE_STEPS" \
  --max_steps "$MAX_STEPS" \
  --output_path "$OUTPUT_DIR" \
  --lora_base_model "dit" \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  "${ANYFLOW_TRAIN_ARGS[@]}" \
  --task "anyflow_stage1:train" \
  2>&1 | tee "$LOG_DIR/train.log"

echo "[OK] AnyFlow Stage 1 train finished. Checkpoints saved every $SAVE_STEPS steps under $OUTPUT_DIR."
