#!/bin/bash
# Train the SEA adapter on DialogRE, initialized from the backbone checkpoint.
# Requires: backbone checkpoint from train_backbone.sh
# Output: ./outputs/sea/sea_seed{SEED}/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while [ ! -d "${ROOT}/data/DialogRE" ] && [ "${ROOT}" != "/" ]; do ROOT="$(dirname "${ROOT}")"; done

SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/DialogRE}"
ROBERTA_DIR="${ROBERTA_DIR:-${ROOT}/pretrained/roberta-base}"
BACKBONE_DIR="${BACKBONE_DIR:-${ROOT}/outputs/backbone/backbone_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/sea/sea_seed${SEED}}"
EVIDENCE_FILE="${EVIDENCE_FILE:-${ROOT}/evidence/dialogre_evidence_full_v2.json}"
GPU="${GPU:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
ADAPTER_LR="${ADAPTER_LR:-1.5e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-6}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
MAX_SUPPORT_TURNS="${MAX_SUPPORT_TURNS:-6}"
EVIDENCE_BIAS_SCALE="${EVIDENCE_BIAS_SCALE:-0.7}"
CONSISTENCY_LOSS_WEIGHT="${CONSISTENCY_LOSS_WEIGHT:-0.02}"
RELATION_PRIOR_WEIGHT="${RELATION_PRIOR_WEIGHT:-0.05}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
    python "${ROOT}/dialogre/run_classifier_sea.py"
    --data_dir "${DATA_DIR}"
    --config_file "${ROBERTA_DIR}/config.json"
    --vocab_file "${ROBERTA_DIR}/vocab.json"
    --merges_file "${ROBERTA_DIR}/merges.txt"
    --output_dir "${OUTPUT_DIR}"
    --data_name DialogRE
    --encoder_type RoBERTa
    --init_checkpoint "${BACKBONE_DIR}/model_best.pt"
    --base_checkpoint "${ROBERTA_DIR}/pytorch_model.bin"
    --do_train --do_eval
    --train_batch_size "${TRAIN_BATCH_SIZE}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --adapter_learning_rate "${ADAPTER_LR}"
    --num_train_epochs "${NUM_EPOCHS}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
    --evidence_file "${EVIDENCE_FILE}"
    --max_support_turns "${MAX_SUPPORT_TURNS}"
    --evidence_bias_scale "${EVIDENCE_BIAS_SCALE}"
    --consistency_loss_weight "${CONSISTENCY_LOSS_WEIGHT}"
    --relation_prior_weight "${RELATION_PRIOR_WEIGHT}"
    --seed "${SEED}"
    --gpu "${GPU}"
)

echo "[SEA] Seed=${SEED}, Backbone=${BACKBONE_DIR}, Output=${OUTPUT_DIR}"
if [ "${DRY_RUN}" = "true" ]; then
    echo "[DRY-RUN] ${CMD[*]}"
else
    "${CMD[@]}"
    echo "[SEA] Done. Best model: ${OUTPUT_DIR}/model_best.pt"
fi
