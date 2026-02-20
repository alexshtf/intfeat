#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

cd "${REPO_ROOT}"

echo "$$" > "${HERE}/pid.txt"
git rev-parse HEAD > "${HERE}/git_commit.txt"
date -Is > "${HERE}/started_at.txt"

TRAIN_ROWS=2000000
VAL_ROWS=400000
TEST_ROWS=400000

TRIALS=100
NUM_EPOCHS=1
BATCH_SIZE=256
EMBEDDING_DIM=8
BSPLINE_KNOTS=10
SL_NUM_BASIS=10

run_one() {
  local name="$1"
  local variant="$2"
  local cfg_flag="$3"
  local cfg_path="$4"
  local study_prefix="$5"

  local out_dir="${HERE}/runs/${name}"
  mkdir -p "${out_dir}"

  echo "===== ${name} (${variant}) =====" | tee "${out_dir}/driver_header.txt" >/dev/null

  # Note: the python runner will also write pid.txt and cmd.txt into out_dir.
  PYTHONUNBUFFERED=1 uv run python -m experiments.criteo_fwfm.optuna_head_contiguous_resumable \
    --train-rows "${TRAIN_ROWS}" --val-rows "${VAL_ROWS}" --test-rows "${TEST_ROWS}" \
    --trials "${TRIALS}" \
    --num-epochs "${NUM_EPOCHS}" --batch-size "${BATCH_SIZE}" --embedding-dim "${EMBEDDING_DIM}" \
    --bspline-knots "${BSPLINE_KNOTS}" --sl-num-basis "${SL_NUM_BASIS}" \
    --variants "${variant}" \
    "${cfg_flag}" "${cfg_path}" \
    --output-json "${out_dir}/results.json" \
    --checkpoint-dir "${out_dir}/checkpoints" \
    --storage-url "sqlite:///${out_dir}/study.sqlite3" \
    --study-prefix "${study_prefix}" \
    > "${out_dir}/run.log" 2>&1
}

run_one \
  "bspline" \
  "bspline_integer_basis" \
  "--bspline-config" \
  "experiments/criteo_fwfm/config/model_bspline_quantile_large_token.yaml" \
  "criteo_head2m_bspline_qcap_large"

run_one \
  "hybrid_u_right" \
  "hybrid_bspline_sl" \
  "--hybrid-config" \
  "experiments/criteo_fwfm/config/model_hybrid_bspline_sl_i5_quantile_large_token_u_right_inverse_square_fixed.yaml" \
  "criteo_head2m_hybrid_i5_u_right_fixed"

run_one \
  "hybrid_inverse_square" \
  "hybrid_bspline_sl" \
  "--hybrid-config" \
  "experiments/criteo_fwfm/config/model_hybrid_bspline_sl_i5_quantile_large_token_inverse_square_fixed.yaml" \
  "criteo_head2m_hybrid_i5_inverse_square_fixed"

date -Is > "${HERE}/finished_at.txt"

