#!/usr/bin/env bash
set -Eeuo pipefail

# AnyFlow/LTX2 8-GPU DeepSpeed ZeRO-3 smoke.
# Data stays on the one-sample smoke metadata/cache, while training hyperparameters
# follow the intended full-training defaults as closely as a smoke run can.

# cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate py312

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN_TAG=${RUN_TAG:-4gpu}
HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-768}
NUM_FRAMES=${NUM_FRAMES:-121}
FRAME_RATE=${FRAME_RATE:-25}

# Full-training style knobs. Keep SAVE_STEPS=1 for smoke so the run emits a checkpoint.
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_STEPS=${SAVE_STEPS:-1}
TRAIN_DATASET_REPEAT=${TRAIN_DATASET_REPEAT:-100}
TRAIN_DATASET_NUM_WORKERS=${TRAIN_DATASET_NUM_WORKERS:-4}
ANYFLOW_DIFFUSION_RATIO=${ANYFLOW_DIFFUSION_RATIO:-0.25}
ANYFLOW_CONSISTENCY_RATIO=${ANYFLOW_CONSISTENCY_RATIO:-0.25}
ANYFLOW_RECONSTRUCTION_WEIGHT=${ANYFLOW_RECONSTRUCTION_WEIGHT:-0.1}

SMOKE_DIR=examples/ltx2_anyflow/smoke/${RUN_TAG}
CACHE_DIR=${CACHE_DIR:-${SMOKE_DIR}/cache_one}
OUTPUT_DIR=${OUTPUT_DIR:-${SMOKE_DIR}/output_one}
LOG_DIR=${LOG_DIR:-${SMOKE_DIR}/logs}
METADATA=${METADATA:-examples/ltx2_anyflow/smoke/metadata_one.csv}
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_DIR"

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

echo "[1/2] Cache smoke data: ${HEIGHT}x${WIDTH}x${NUM_FRAMES} -> $CACHE_DIR"
accelerate launch examples/ltx2_anyflow/model_training/train.py \
  --dataset_base_path "" \
  --dataset_metadata_path "$METADATA" \
  "${COMMON_MODEL_ARGS[@]}" \
  --dataset_repeat 1 \
  --dataset_num_workers 0 \
  --model_id_with_origin_paths "$ENCODER_MODEL_SPEC" \
  --learning_rate "$LEARNING_RATE" \
  --num_epochs 1 \
  --output_path "$CACHE_DIR" \
  --task "flowmap_sft:data_process" \
  2>&1 | tee "$LOG_DIR/data_process.log"

echo "[2/2] 8-GPU ZeRO-3 smoke train with full-style hyperparams -> $OUTPUT_DIR"
echo "lr=$LEARNING_RATE weight_decay=$WEIGHT_DECAY epochs=$NUM_EPOCHS repeat=$TRAIN_DATASET_REPEAT recon=$ANYFLOW_RECONSTRUCTION_WEIGHT"
accelerate launch --config_file "$ACCELERATE_CONFIG" examples/ltx2_anyflow/model_training/train.py \
  --dataset_base_path "$CACHE_DIR" \
  "${COMMON_MODEL_ARGS[@]}" \
  --dataset_repeat "$TRAIN_DATASET_REPEAT" \
  --dataset_num_workers "$TRAIN_DATASET_NUM_WORKERS" \
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

echo "Smoke finished. Logs: $LOG_DIR"
