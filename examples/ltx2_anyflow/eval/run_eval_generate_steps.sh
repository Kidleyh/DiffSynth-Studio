#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py312

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

VAL_METADATA=${VAL_METADATA:-data/val.csv}
OUTPUT_DIR=${OUTPUT_DIR:-examples/ltx2_anyflow/eval/generated_steps}
TRAINED_CKPT=${TRAINED_CKPT:-examples/ltx2_anyflow/smoke/4gpu/output_one/step-1.safetensors}
BASE_TRANSFORMER_SPEC=${BASE_TRANSFORMER_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors}
STEPS=${STEPS:-10,15,20,30,40}
NUM_SAMPLES=${NUM_SAMPLES:-4}
SEED=${SEED:-43}
HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-768}
NUM_FRAMES=${NUM_FRAMES:-121}
FPS=${FPS:-25}

python examples/ltx2_anyflow/eval_generate_steps.py \
  --metadata_path "$VAL_METADATA" \
  --output_dir "$OUTPUT_DIR" \
  --base_transformer_spec "$BASE_TRANSFORMER_SPEC" \
  --checkpoints "base=BASE" "trained=$TRAINED_CKPT" \
  --steps "$STEPS" \
  --num_samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --fps "$FPS"
