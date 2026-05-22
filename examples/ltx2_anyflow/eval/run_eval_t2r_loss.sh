#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py312

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

VAL_METADATA=${VAL_METADATA:-data/val.csv}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-examples/ltx2_anyflow/eval/cache_val}
EVAL_OUTPUT_CSV=${EVAL_OUTPUT_CSV:-examples/ltx2_anyflow/eval/t2r_loss.csv}
TRAINED_CKPT=${TRAINED_CKPT:-examples/ltx2_anyflow/smoke/4gpu/output_one/step-1.safetensors}
MAX_SAMPLES=${MAX_SAMPLES:-16}
HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-768}
NUM_FRAMES=${NUM_FRAMES:-121}
FRAME_RATE=${FRAME_RATE:-25}

ENCODER_MODEL_SPEC=${ENCODER_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors}
BASE_TRANSFORMER_SPEC=${BASE_TRANSFORMER_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors}

mkdir -p "$EVAL_CACHE_DIR" "$(dirname "$EVAL_OUTPUT_CSV")"

if ! find "$EVAL_CACHE_DIR" -name '*.pth' -print -quit | grep -q .; then
  echo "[1/2] Build eval cache from $VAL_METADATA -> $EVAL_CACHE_DIR"
  accelerate launch examples/ltx2_anyflow/model_training/train.py \
    --dataset_base_path "" \
    --dataset_metadata_path "$VAL_METADATA" \
    --data_file_keys "video,input_audio" \
    --extra_inputs "input_audio,input_image" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_frames "$NUM_FRAMES" \
    --frame_rate "$FRAME_RATE" \
    --dataset_repeat 1 \
    --dataset_num_workers 0 \
    --trainable_models "dit" \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --model_id_with_origin_paths "$ENCODER_MODEL_SPEC" \
    --learning_rate 1e-6 \
    --num_epochs 1 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "$EVAL_CACHE_DIR" \
    --task "flowmap_sft:data_process"
else
  echo "[1/2] Reuse eval cache: $EVAL_CACHE_DIR"
fi

echo "[2/2] Evaluate t->r loss base vs trained"
python examples/ltx2_anyflow/eval_t2r_loss.py \
  --cache_dir "$EVAL_CACHE_DIR" \
  --output_csv "$EVAL_OUTPUT_CSV" \
  --base_transformer_spec "$BASE_TRANSFORMER_SPEC" \
  --checkpoints "base=BASE" "trained=$TRAINED_CKPT" \
  --max_samples "$MAX_SAMPLES" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --frame_rate "$FRAME_RATE"
