#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py312

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN_NAME=${RUN_NAME:-ltx2_anyflow_0430}
DATASET_BASE_PATH=${DATASET_BASE_PATH:-""}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-data/training_data_all.csv}
CACHE_DIR=${CACHE_DIR:-examples/ltx2_anyflow/cache/${RUN_NAME}}
OUTPUT_DIR=${OUTPUT_DIR:-examples/ltx2_anyflow/output/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-examples/ltx2_anyflow/logs/${RUN_NAME}}
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_DIR"

HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-768}
NUM_FRAMES=${NUM_FRAMES:-121}
FRAME_RATE=${FRAME_RATE:-25}
DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-4}
TRAIN_DATASET_REPEAT=${TRAIN_DATASET_REPEAT:-1}

LEARNING_RATE=${LEARNING_RATE:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_STEPS=${SAVE_STEPS:-50000}
ANYFLOW_DIFFUSION_RATIO=${ANYFLOW_DIFFUSION_RATIO:-0.25}
ANYFLOW_CONSISTENCY_RATIO=${ANYFLOW_CONSISTENCY_RATIO:-0.25}
ANYFLOW_RECONSTRUCTION_WEIGHT=${ANYFLOW_RECONSTRUCTION_WEIGHT:-0.1}

ENCODER_MODEL_SPEC=${ENCODER_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors}
TRANSFORMER_MODEL_SPEC=${TRANSFORMER_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-examples/ltx2_anyflow/configs/accelerate_config_zero3_8gpu.yaml}

COMMON_MODEL_ARGS=(
  --data_file_keys "video,input_audio"
  --extra_inputs "input_audio,input_image"
  --height "$HEIGHT"
  --width "$WIDTH"
  --num_frames "$NUM_FRAMES"
  --frame_rate "$FRAME_RATE"
  --trainable_models "dit"
  --use_gradient_checkpointing
  --find_unused_parameters
  --weight_decay "$WEIGHT_DECAY"
  --remove_prefix_in_ckpt "pipe.dit."
)

echo "[1/2] Cache full data -> $CACHE_DIR"
accelerate launch examples/ltx2_anyflow/model_training/train.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATASET_METADATA_PATH" \
  "${COMMON_MODEL_ARGS[@]}" \
  --dataset_repeat 1 \
  --dataset_num_workers "$DATASET_NUM_WORKERS" \
  --model_id_with_origin_paths "$ENCODER_MODEL_SPEC" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs 1 \
  --output_path "$CACHE_DIR" \
  --task "flowmap_sft:data_process" \
  2>&1 | tee "$LOG_DIR/data_process.log"

echo "[2/2] 8-GPU ZeRO-3 train -> $OUTPUT_DIR"
echo "metadata=$DATASET_METADATA_PATH lr=$LEARNING_RATE weight_decay=$WEIGHT_DECAY epochs=$NUM_EPOCHS repeat=$TRAIN_DATASET_REPEAT recon=$ANYFLOW_RECONSTRUCTION_WEIGHT"
accelerate launch --config_file "$ACCELERATE_CONFIG" examples/ltx2_anyflow/model_training/train.py \
  --dataset_base_path "$CACHE_DIR" \
  "${COMMON_MODEL_ARGS[@]}" \
  --dataset_repeat "$TRAIN_DATASET_REPEAT" \
  --dataset_num_workers "$DATASET_NUM_WORKERS" \
  --model_id_with_origin_paths "$TRANSFORMER_MODEL_SPEC" \
  --initialize_model_on_cpu \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs "$NUM_EPOCHS" \
  --save_steps "$SAVE_STEPS" \
  --output_path "$OUTPUT_DIR" \
  --task "flowmap_sft:train" \
  --anyflow_diffusion_ratio "$ANYFLOW_DIFFUSION_RATIO" \
  --anyflow_consistency_ratio "$ANYFLOW_CONSISTENCY_RATIO" \
  --anyflow_reconstruction_weight "$ANYFLOW_RECONSTRUCTION_WEIGHT" \
  2>&1 | tee "$LOG_DIR/train.log"

echo "Training finished. Logs: $LOG_DIR"
