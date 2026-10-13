#!/bin/bash
# Train the HiDialog-GTN backbone on DialogRE.
# Output: ./outputs/backbone/backbone_seed{SEED}/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while [ ! -d "${ROOT}/data/DialogRE" ] && [ "${ROOT}" != "/" ]; do ROOT="$(dirname "${ROOT}")"; done

SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/DialogRE}"
ROBERTA_DIR="${ROBERTA_DIR:-${ROOT}/pretrained/roberta-base}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/backbone/backbone_seed${SEED}}"
GPU="${GPU:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
    python "${ROOT}/dialogre/run_classifier.py"
    --data_dir "${DATA_DIR}"
    --config_file "${ROBERTA_DIR}/config.json"
    --vocab_file "${ROBERTA_DIR}/vocab.json"
    --merges_file "${ROBERTA_DIR}/merges.txt"
    --output_dir "${OUTPUT_DIR}"
    --data_name DialogRE
    --encoder_type RoBERTa
    --do_train --do_eval
    --train_batch_size "${TRAIN_BATCH_SIZE}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --num_train_epochs "${NUM_EPOCHS}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
    --seed "${SEED}"
    --gpu "${GPU}"
)

echo "[Backbone] Seed=${SEED}, Output=${OUTPUT_DIR}"
if [ "${DRY_RUN}" = "true" ]; then
    echo "[DRY-RUN] ${CMD[*]}"
else
    "${CMD[@]}"
    echo "[Backbone] Done. Best model: ${OUTPUT_DIR}/model_best.pt"
fi
