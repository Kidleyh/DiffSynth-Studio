#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio
source /root/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth_ltx

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN_NAME=${RUN_NAME:-LTX2.3-I2AV-lyh-smoke4}
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
  --use_gradient_checkpointing
  --remove_prefix_in_ckpt "pipe.dit."
)

echo "[1/2] Native LTX2 data_process smoke -> $CACHE_DIR"
accelerate launch --num_processes 1 examples/ltx2/model_training/train.py \
  --dataset_base_path "" \
  --dataset_metadata_path "$METADATA" \
  "${COMMON_ARGS[@]}" \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "$ENCODER_MODEL_SPEC" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --output_path "$CACHE_DIR" \
  --task "sft:data_process" \
  2>&1 | tee "$LOG_DIR/data_process.log"

echo "[2/2] Native LTX2 4-GPU ZeRO3 train smoke -> $OUTPUT_DIR"
accelerate launch --config_file "$ACCELERATE_CONFIG" examples/ltx2/model_training/train.py \
  --dataset_base_path "$CACHE_DIR" \
  "${COMMON_ARGS[@]}" \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "$TRANSFORMER_MODEL_SPEC" \
  --initialize_model_on_cpu \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --save_steps 1 \
  --output_path "$OUTPUT_DIR" \
  --task "sft:train" \
  2>&1 | tee "$LOG_DIR/train.log"

echo "Smoke finished. Logs: $LOG_DIR"
