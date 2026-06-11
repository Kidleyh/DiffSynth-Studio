#!/usr/bin/env bash
set -Eeuo pipefail

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

RUN_NAME=${RUN_NAME:-LTX2.3-I2AV-anyflow-stage1-lowres-train1-test}
OUTPUT_DIR=${OUTPUT_DIR:-./models/train/${RUN_NAME}}
CACHE_DIR=${CACHE_DIR:-./models/train/${RUN_NAME}-cache}
LOG_DIR=${LOG_DIR:-./models/train/${RUN_NAME}-logs}
SAMPLE_MODEL_SPEC=${SAMPLE_MODEL_SPEC:-DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors}
MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH:-}

HEIGHT=${HEIGHT:-128}
WIDTH=${WIDTH:-128}
NUM_FRAMES=${NUM_FRAMES:-9}
MAX_STEPS=${MAX_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-1}
TRAIN_DATASET_REPEAT=${TRAIN_DATASET_REPEAT:-1}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-8}
PROMPT=${PROMPT:-a dog running on the grass, natural sound}

python -m py_compile examples/ltx2_anyflow_stage1/*.py

mkdir -p "$LOG_DIR"
if [[ -z "$MODEL_CONFIG_PATH" ]]; then
  MODEL_CONFIG_PATH="${LOG_DIR}/sample_model_config.json"
  SAMPLE_MODEL_SPEC="$SAMPLE_MODEL_SPEC" MODEL_CONFIG_PATH="$MODEL_CONFIG_PATH" python - <<'PYMODEL'
import json
import os
from pathlib import Path

items = []
for spec in os.environ["SAMPLE_MODEL_SPEC"].split(","):
    spec = spec.strip()
    if not spec:
        continue
    if ":" not in spec:
        raise SystemExit(f"Invalid SAMPLE_MODEL_SPEC item: {spec!r}")
    model_id, pattern = spec.rsplit(":", 1)
    items.append({"model_id": model_id, "origin_file_pattern": pattern})
Path(os.environ["MODEL_CONFIG_PATH"]).write_text(json.dumps({"model_configs": items}, indent=2))
print(f"[INFO] Generated sampler model config: {os.environ['MODEL_CONFIG_PATH']}")
PYMODEL
else
  echo "[INFO] Using provided MODEL_CONFIG_PATH=$MODEL_CONFIG_PATH"
fi

LOW_RES_SMOKE=1 \
HEIGHT="$HEIGHT" \
WIDTH="$WIDTH" \
NUM_FRAMES="$NUM_FRAMES" \
MAX_STEPS="$MAX_STEPS" \
SAVE_STEPS="$SAVE_STEPS" \
TRAIN_DATASET_REPEAT="$TRAIN_DATASET_REPEAT" \
RUN_NAME="$RUN_NAME" \
CACHE_DIR="$CACHE_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
LOG_DIR="$LOG_DIR" \
bash examples/ltx2_anyflow_stage1/LTX-2.3-I2AV-anyflow-stage1-lora-smoke4_4gpu.sh

CKPT="${OUTPUT_DIR}/checkpoint-step_000001"
test -f "${CKPT}/anyflow_wrapper.pt"
test -f "${CKPT}/anyflow_config.json"
test -f "${OUTPUT_DIR}/gradient_sanity_step_000001.json"
test -f "${OUTPUT_DIR}/native_key_report_step_000001.json"
test -f "${OUTPUT_DIR}/anyflow_stage1_log_step_000001.json"

python - <<PY
import json
import math
from pathlib import Path

ckpt = Path("${CKPT}")
out = Path("${OUTPUT_DIR}")
cfg = json.loads((ckpt / "anyflow_config.json").read_text())
grad = json.loads((out / "gradient_sanity_step_000001.json").read_text())
log = json.loads((out / "anyflow_stage1_log_step_000001.json").read_text())

if not (cfg.get("use_lora") is True or cfg.get("lora_base_model") == "dit"):
    raise SystemExit("checkpoint config does not indicate DiT LoRA")
if int(cfg.get("lora_rank", -1)) != 8:
    raise SystemExit(f"expected lora_rank=8, got {cfg.get('lora_rank')!r}")
if int(grad.get("trainable_without_grad_count", -1)) != 0:
    raise SystemExit(f"trainable_without_grad_count is not 0: {grad.get('trainable_without_grad_count')!r}")
loss_total = float(log.get("loss_total"))
if not math.isfinite(loss_total):
    raise SystemExit(f"loss_total is not finite: {loss_total!r}")
print(f"[CHECK] config LoRA ok: base={cfg.get('lora_base_model')} rank={cfg.get('lora_rank')} alpha={cfg.get('lora_alpha')}")
print(f"[CHECK] gradient sanity ok: trainable_without_grad_count={grad.get('trainable_without_grad_count')}")
print(f"[CHECK] loss_total finite: {loss_total}")
print(f"[CHECK] audio_present={log.get('audio_present', 'missing')}")
PY

ROLLOUT_DIR="${OUTPUT_DIR}/latent_rollout_4step"
python examples/ltx2_anyflow_stage1/sample_ltx2_anyflow_stage1.py \
  --checkpoint "$CKPT" \
  --model_config_path "$MODEL_CONFIG_PATH" \
  --prompt "$PROMPT" \
  --num_inference_steps 4 \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --latent_rollout_only \
  --output_path "$ROLLOUT_DIR"

test -d "$ROLLOUT_DIR"
test -f "${ROLLOUT_DIR}/rollout_stats.json"
test -f "${ROLLOUT_DIR}/final_video_latents.pt"
test -f "${ROLLOUT_DIR}/final_audio_latents.pt"

python - <<PY
import json
from pathlib import Path
stats = json.loads((Path("${ROLLOUT_DIR}") / "rollout_stats.json").read_text())
if int(stats.get("num_inference_steps", -1)) != 4:
    raise SystemExit(f"expected 4 rollout steps, got {stats.get('num_inference_steps')!r}")
if len(stats.get("steps", [])) != 4:
    raise SystemExit(f"expected 4 per-step stats, got {len(stats.get('steps', []))}")
print(f"[CHECK] latent rollout ok: steps={stats.get('num_inference_steps')} final_video_norm={stats.get('final_video_norm')}")
PY

echo "[OK] Low-res 4-GPU train-1-step + native checkpoint latent sampling test passed."
