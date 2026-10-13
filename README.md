# SEA on DialogRE

Structured Evidence Aggregation (SEA) for dialogue-based relation extraction on the DialogRE benchmark. The model builds on a HiDialog-GTN backbone and aggregates structured evidence (supporting turns, turn-level confidences, relation priors) with a confidence gate, consistency loss, and relation prior regularization.

## 1. Install

Linux with an NVIDIA GPU and Python 3.10 is recommended.

```bash
git clone https://github.com/<your-account>/SEA.git
cd SEA
conda create -n sea python=3.10 -y
conda activate sea
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Download Data and Pretrained Model

- DialogRE dataset: follow `data/DialogRE/README.md`.
- RoBERTa-base: follow `pretrained/README.md`.

No data or model files are bundled in this repository.

## 3. Train

Stage 1 - train the HiDialog-GTN backbone:

```bash
bash scripts/train_backbone.sh
```

Stage 2 - train the SEA adapter on top of the backbone checkpoint:

```bash
bash scripts/train_sea.sh
```

Change the random seed with `SEED=123 bash scripts/train_sea.sh`, or override other hyper-parameters via the environment variables defined at the top of each script.

Small CPU test:

```bash
export GPU=-1
DRY_RUN=true bash scripts/train_backbone.sh
```

## 4. Evaluate

Each training script runs evaluation on the dev split with `--do_eval`. To evaluate an existing checkpoint only:

```bash
python dialogre/run_classifier_sea.py \
  --data_dir data/DialogRE \
  --config_file pretrained/roberta-base/config.json \
  --vocab_file pretrained/roberta-base/vocab.json \
  --merges_file pretrained/roberta-base/merges.txt \
  --output_dir outputs/sea/sea_seed42 \
  --data_name DialogRE \
  --encoder_type RoBERTa \
  --do_eval \
  --init_checkpoint outputs/backbone/backbone_seed42/model_best.pt \
  --base_checkpoint pretrained/roberta-base/pytorch_model.bin \
  --evidence_file evidence/dialogre_evidence_full_v2.json
```

Results are saved under `outputs/`.

## Reference

- DialogRE: [Dialogue-Based Relation Extraction](https://aclanthology.org/2021.emnlp-main.402/), Yu et al., EMNLP 2021.
