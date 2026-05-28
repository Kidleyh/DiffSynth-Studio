#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN_NAME=${RUN_NAME:-LTX2.3-I2AV-anyflow-stage1-lora-smoke4}
METADATA=${METADATA:-examples/ltx2/model_training/full/metadata_lyh_smoke4.csv}
CACHE_DIR=${CACHE_DIR:-./models/train/${RUN_NAME}-cache}
OUTPUT_DIR=${OUTPUT_DIR:-./models/train/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-./models/train/${RUN_NAME}-logs}
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_DIR"

HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-768}
NUM_FRAMES=${NUM_FRAMES:-121}
FRAME_RATE=${FRAME_RATE:-25}
DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-0}
TRAIN_DATASET_REPEAT=${TRAIN_DATASET_REPEAT:-25}
MAX_STEPS=${MAX_STEPS:-2}
SAVE_STEPS=${SAVE_STEPS:-2}
USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-1}

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

echo "[1/2] AnyFlow native data_process smoke -> $CACHE_DIR"
accelerate launch --num_processes 1 examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py \
  --dataset_base_path "" \
  --dataset_metadata_path "$METADATA" \
  "${COMMON_ARGS[@]}" \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "$ENCODER_MODEL_SPEC" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --output_path "$CACHE_DIR" \
  --task "anyflow_stage1:data_process" \
  2>&1 | tee "$LOG_DIR/data_process.log"

echo "[2/2] AnyFlow native 4-GPU LoRA train smoke -> $OUTPUT_DIR"
accelerate launch --config_file "$ACCELERATE_CONFIG" examples/ltx2_anyflow_stage1/train_ltx2_anyflow_stage1_native.py \
  --dataset_base_path "$CACHE_DIR" \
  "${COMMON_ARGS[@]}" \
  --dataset_repeat "$TRAIN_DATASET_REPEAT" \
  --model_id_with_origin_paths "$TRANSFORMER_MODEL_SPEC" \
  --initialize_model_on_cpu \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --save_steps "$SAVE_STEPS" \
  --max_steps "$MAX_STEPS" \
  --output_path "$OUTPUT_DIR" \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_q,to_v,to_out.0" \
  --lora_rank 8 \
  --lora_alpha 8 \
  --save_gradient_sanity \
  --task "anyflow_stage1:train" \
  2>&1 | tee "$LOG_DIR/train.log"

echo "Smoke finished. Logs: $LOG_DIR"
echo "For formal training, use --lora_rank 256 --lora_alpha 256 and increase MAX_STEPS/SAVE_STEPS."
