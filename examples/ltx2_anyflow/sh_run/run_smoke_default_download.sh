#!/usr/bin/env bash
set -Eeuo pipefail

# One-sample AnyFlow/LTX2 smoke test.
# This version uses DiffSynth/ModelScope specs and will download missing weights if they are not cached.

cd /gemini/platform/public/aigc/human_guozz2/code/lyh/job/DiffSynth-Studio
source /root/miniconda3/etc/profile.d/conda.sh
conda activate far

bash examples/ltx2_anyflow/smoke/run_one_sample.sh
