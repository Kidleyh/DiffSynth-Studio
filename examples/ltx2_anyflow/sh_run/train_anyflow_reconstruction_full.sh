#!/usr/bin/env bash
set -Eeuo pipefail

# Full two-stage LTX2 AnyFlow training with flow-map reconstruction loss enabled.
# Stage 1 caches encoder/text/audio/video features from the CSV metadata.
# Stage 2 trains the DiT with FlowMapSFTAudioVideoLoss, including the explicit t -> r reconstruction objective.

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio
source /root/miniconda3/etc/profile.d/conda.sh
conda activate far

DATASET_BASE_PATH=${DATASET_BASE_PATH:-""}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-/gemini/platform/public/aigc/human_guozz2/code/zhangyan/DiffSynth-Studio-LTX/data/training_data_all.csv}

RUN_NAME=${RUN_NAME:-ltx2_anyflow_reconstruction}
CACHE_DIR=${CACHE_DIR:-examples/ltx2_anyflow/cache/${RUN_NAME}}
OUTPUT_DIR=${OUTPUT_DIR:-examples/ltx2_anyflow/output/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-examples/ltx2_anyflow/logs/${RUN_NAME}}
mkdir -p "$LOG_DIR"

# LTX2 requires num_frames % 8 == 1. For real training, use 121 if memory allows.
HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-768}
NUM_FRAMES=${NUM_FRAMES:-121}
FRAME_RATE=${FRAME_RATE:-25}
DATASET_REPEAT=${DATASET_REPEAT:-1}
DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-4}

LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_STEPS=${SAVE_STEPS:-1000}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}

ANYFLOW_DIFFUSION_RATIO=${ANYFLOW_DIFFUSION_RATIO:-0.25}
ANYFLOW_CONSISTENCY_RATIO=${ANYFLOW_CONSISTENCY_RATIO:-0.25}
ANYFLOW_RECONSTRUCTION_WEIGHT=${ANYFLOW_RECONSTRUCTION_WEIGHT:-0.1}

# Stage 1 encoder/tokenizer specs. These download from ModelScope/HF if not already cached.
ENCODER_MODEL_SPEC=${ENCODER_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors}

# Stage 2 transformer spec. Override this env var to use a local checkpoint, for example:
# TRANSFORMER_MODEL_SPEC=/gemini/platform/public/aigc/human_guozz2/code/zhangyan/DiffSynth-Studio-LTX/model_output/LTX2.3-I2AV-moe-0519/step-3000.safetensors
TRANSFORMER_MODEL_SPEC=${TRANSFORMER_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors}

COMMON_ARGS=(
  --data_file_keys "video,input_audio"
  --extra_inputs "input_audio,input_image"
  --height "$HEIGHT"
  --width "$WIDTH"
  --num_frames "$NUM_FRAMES"
  --frame_rate "$FRAME_RATE"
  --dataset_repeat "$DATASET_REPEAT"
  --dataset_num_workers "$DATASET_NUM_WORKERS"
  --trainable_models "dit"
  --use_gradient_checkpointing
  --find_unused_parameters
  --weight_decay "$WEIGHT_DECAY"
  --remove_prefix_in_ckpt "pipe.dit."
)

echo "[1/2] flowmap_sft:data_process -> $CACHE_DIR"
accelerate launch examples/ltx2_anyflow/model_training/train.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATASET_METADATA_PATH" \
  "${COMMON_ARGS[@]}" \
  --model_id_with_origin_paths "$ENCODER_MODEL_SPEC" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs 1 \
  --output_path "$CACHE_DIR" \
  --task "flowmap_sft:data_process" \
  2>&1 | tee "$LOG_DIR/data_process.log"

echo "[2/2] flowmap_sft:train with reconstruction weight=$ANYFLOW_RECONSTRUCTION_WEIGHT -> $OUTPUT_DIR"
accelerate launch examples/ltx2_anyflow/model_training/train.py \
  --dataset_base_path "$CACHE_DIR" \
  "${COMMON_ARGS[@]}" \
  --model_id_with_origin_paths "$TRANSFORMER_MODEL_SPEC" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs "$NUM_EPOCHS" \
  --save_steps "$SAVE_STEPS" \
  --output_path "$OUTPUT_DIR" \
  --task "flowmap_sft:train" \
  --anyflow_diffusion_ratio "$ANYFLOW_DIFFUSION_RATIO" \
  --anyflow_consistency_ratio "$ANYFLOW_CONSISTENCY_RATIO" \
  --anyflow_reconstruction_weight "$ANYFLOW_RECONSTRUCTION_WEIGHT" \
  2>&1 | tee "$LOG_DIR/train.log"
