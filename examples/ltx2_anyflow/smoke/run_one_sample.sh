#!/usr/bin/env bash
set -Eeuo pipefail

# One-sample smoke test for examples/ltx2_anyflow.
# It mirrors DiffSynth's LTX2 split-training flow:
#   1) flowmap_sft:data_process caches text/audio/video encodings.
#   2) flowmap_sft:train loads the cached tensors and runs one AnyFlow loss/backward step.

REPO_ROOT=${REPO_ROOT:-/gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio}
cd "$REPO_ROOT"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-far}"

SMOKE_DIR=${SMOKE_DIR:-examples/ltx2_anyflow/smoke}
METADATA=${METADATA:-$SMOKE_DIR/metadata_one.csv}
CACHE_DIR=${CACHE_DIR:-$SMOKE_DIR/cache_one}
OUTPUT_DIR=${OUTPUT_DIR:-$SMOKE_DIR/output_one}
LOG_DIR=${LOG_DIR:-$SMOKE_DIR/logs}
mkdir -p "$SMOKE_DIR" "$LOG_DIR"
rm -rf "$CACHE_DIR" "$OUTPUT_DIR"

# Small shape for code validation. LTX2 requires num_frames % 8 == 1.
HEIGHT=${HEIGHT:-256}
WIDTH=${WIDTH:-384}
NUM_FRAMES=${NUM_FRAMES:-9}
FRAME_RATE=${FRAME_RATE:-25}

# Stage 1 needs tokenizer/text post modules and VAE encoders.
ENCODER_MODEL_SPEC=${ENCODER_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors}

# Stage 2 needs only the DiT/transformer. You can override this with a local checkpoint, e.g.:
#   TRANSFORMER_MODEL_SPEC=/gemini/.../model_output/LTX2.3-I2AV-moe-0519/step-3000.safetensors bash ...
TRANSFORMER_MODEL_SPEC=${TRANSFORMER_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors}

COMMON_DATA_ARGS=(
  --data_file_keys "video,input_audio"
  --extra_inputs "input_audio,input_image"
  --height "$HEIGHT"
  --width "$WIDTH"
  --num_frames "$NUM_FRAMES"
  --frame_rate "$FRAME_RATE"
  --dataset_repeat 1
  --dataset_num_workers 0
  --trainable_models "dit"
  --use_gradient_checkpointing
  --find_unused_parameters
)

echo "[1/2] Caching one sample to $CACHE_DIR"
accelerate launch examples/ltx2_anyflow/model_training/train.py \
  --dataset_base_path "" \
  --dataset_metadata_path "$METADATA" \
  "${COMMON_DATA_ARGS[@]}" \
  --model_id_with_origin_paths "$ENCODER_MODEL_SPEC" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$CACHE_DIR" \
  --task "flowmap_sft:data_process" \
  2>&1 | tee "$LOG_DIR/data_process.log"

echo "[2/2] Running one AnyFlow training step to $OUTPUT_DIR"
accelerate launch examples/ltx2_anyflow/model_training/train.py \
  --dataset_base_path "$CACHE_DIR" \
  "${COMMON_DATA_ARGS[@]}" \
  --model_id_with_origin_paths "$TRANSFORMER_MODEL_SPEC" \
  --learning_rate 1e-6 \
  --num_epochs 1 \
  --save_steps 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_DIR" \
  --task "flowmap_sft:train" \
  --anyflow_diffusion_ratio 0.25 \
  --anyflow_consistency_ratio 0.25 \
  --anyflow_reconstruction_weight 0.1 \
  2>&1 | tee "$LOG_DIR/train.log"

echo "Smoke test finished. Outputs:"
find "$SMOKE_DIR" -maxdepth 3 -type f | sort
