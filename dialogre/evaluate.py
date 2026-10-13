
import json
import logging
import os

import numpy as np

from core.evaluation import sigmoid, load_logits

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def getpred(result, threshold1=0.5, threshold2=0.4):
    predictions = []
    for row in result:
        active = []
        max_score, max_label = -1, -1

        for label_idx, score in enumerate(row):
            if score > threshold1:
                active.append(label_idx)
            if score > max_score:
                max_score = score
                max_label = label_idx

        if not active:
            if max_score <= threshold2:
                active = [36]
            else:
                active = [max_label]

        predictions.append(active)

    return predictions


def geteval(predictions, gold_labels):
    correct_sys = 0
    all_sys = 0
    correct_gt = 0

    for idx in range(len(gold_labels)):
        for label_id in gold_labels[idx]:
            if label_id != 36:
                correct_gt += 1
                if label_id in predictions[idx]:
                    correct_sys += 1

        for label_id in predictions[idx]:
            if label_id != 36:
                all_sys += 1

    precision = 1 if all_sys == 0 else correct_sys / all_sys
    recall = 0 if correct_gt == 0 else correct_sys / correct_gt

    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def f1_eval(logits, labels_all):
    logits = np.asarray(logits)
    logits = sigmoid(logits)

    labels = []
    for label_row in labels_all:
        row = np.asarray(label_row).reshape(-1)
        if row.size > 1 and np.all(np.isin(row, [0, 1])):
            active = [idx for idx in range(row.size) if row[idx] == 1]
            labels.append(active if active else [36])
        else:
            active = [int(x) for x in row]
            labels.append(active if active else [36])

    best_threshold2 = 0.0
    best_f1 = 0.0

    for threshold2 in range(51):
        predictions = getpred(logits, threshold2=threshold2 / 100.0)
        score = geteval(predictions, labels)
        if score > best_f1:
            best_f1 = score
            best_threshold2 = threshold2 / 100.0

    return best_f1, best_threshold2


def load_gold_labels(data_dir, split="dev"):
    from data import DialogREProcessor

    processor = DialogREProcessor(
        src_file=data_dir,
        n_class=36,
        dataset="DialogRE",
        conversational_f1=split in ["devc", "testc"],
    )

    if split == "train":
        examples = processor.get_train_examples()
    elif split in ["dev", "devc"]:
        examples = processor.get_dev_examples()
    elif split in ["test", "testc"]:
        examples = processor.get_test_examples()
    else:
        raise ValueError(f"Unknown split: {split}")

    gold_labels = []
    for example in examples:
        active = [idx for idx in range(36) if example.label[idx] == 1]
        gold_labels.append(active if active else [36])

    return gold_labels


def evaluate_predictions(logits_path, data_dir, split="dev", threshold2=None):
    logits = load_logits(logits_path)
    gold_labels = load_gold_labels(data_dir, split)

    if threshold2 is not None:
        predictions = getpred(logits, threshold2=threshold2)
        f1 = geteval(predictions, gold_labels)
        return {"f1": f1, "threshold2": threshold2}

    f1, best_threshold2 = f1_eval(logits, gold_labels)
    return {"f1": f1, "threshold2": best_threshold2}


def write_predictions(logits, output_path, threshold1=0.5, threshold2=0.4):
    predictions = getpred(logits, threshold1=threshold1, threshold2=threshold2)

    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(",".join(str(p) for p in pred) + "\n")


class DialogREEvaluator:

    def __init__(self, data_dir, num_relations=36):
        self.data_dir = data_dir
        self.num_relations = num_relations
        self.best_threshold2 = None

    def compute_f1(self, logits, labels, search_threshold=True):
        if search_threshold:
            return f1_eval(logits, labels)
        else:
            predictions = getpred(logits, threshold2=self.best_threshold2 or 0.4)
            gold_labels = []
            for label_row in labels:
                active = [idx for idx in range(36) if label_row[idx] == 1]
                gold_labels.append(active if active else [36])
            return geteval(predictions, gold_labels), self.best_threshold2 or 0.4


def evaluate_model(
    checkpoint: str = None,
    data_dir: str = None,
    evidence_file: str = None,
    batch_size: int = 8,
    device: str = "cuda",
    logits_path: str = None,
    gold_path: str = None,
) -> dict:
    if logits_path and gold_path:
        predictions = load_logits(logits_path)
        gold_labels = load_gold_labels(gold_path, split="test")
        f1, t2 = f1_eval(predictions, gold_labels)
        return {"test": {"f1": f1, "T2": t2}}
    elif checkpoint and data_dir and evidence_file:
        try:
            import torch
        except ImportError:
            logger.error("torch required for model-based evaluation")
            return {"dev": {"f1": 0.0}, "test": {"f1": 0.0}}

        device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading checkpoint from {checkpoint} on {device}")

        try:
            import os as _os
            from data_structured_evidence import (
                HiDialogDatasetStructuredEvidence,
                HiDialogDataloaderStructuredEvidence,
            )
            from models.RoBERTa.HiDialog_RoBERTa_SEA import HiDialog_RoBERTa_SEA
            from models.RoBERTa.configuration_roberta import RobertaConfig
            from models.RoBERTa.tokenization_roberta import RobertaTokenizer
        except ImportError as e:
            logger.warning(f"Could not import model modules: {e}")
            return {"dev": {"f1": 0.0, "T2": 0.0}, "test": {"f1": 0.0, "T2": 0.0}}

        _proj = _os.path.dirname(_os.path.abspath(__file__))
        _proj = _os.path.dirname(_proj)
        _pt_dir = _os.path.join(_proj, "pretrained", "roberta-base")
        config = RobertaConfig.from_json_file(_os.path.join(_pt_dir, "config.json"))
        tokenizer = RobertaTokenizer(
            vocab_file=_os.path.join(_pt_dir, "vocab.json"),
            merges_file=_os.path.join(_pt_dir, "merges.txt"),
        )
        tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]"]})

        model = HiDialog_RoBERTa_SEA(config, num_labels=36, dataset="DialogRE")
        model.roberta.resize_token_embeddings(tokenizer.vocab_size)
        state = torch.load(checkpoint, map_location="cpu")
        model_state = model.state_dict()
        load_dict = {}
        for k, v in state.items():
            if k not in model_state:
                continue
            t = model_state[k]
            if v.shape == t.shape:
                load_dict[k] = v
            elif v.dim() == 2 and v.shape[1] == t.shape[1] and t.shape[0] > v.shape[0]:
                merged = t.clone()
                merged[: v.shape[0]] = v
                load_dict[k] = merged
        model.load_state_dict(load_dict, strict=False)
        model.to(device)
        model.eval()

        results = {}
        for split in ["dev", "test"]:
            cache_file = _os.path.join(
                data_dir,
                f"{split}_RoBERTa_sea_{_os.path.splitext(_os.path.basename(evidence_file))[0]}_ms6_seq512.pkl",
            )
            dataset = HiDialogDatasetStructuredEvidence(
                src_file=data_dir,
                save_file=cache_file,
                max_seq_length=512,
                tokenizer=tokenizer,
                n_class=36,
                encoder_type="RoBERTa",
                dataset="DialogRE",
                split_name=split,
                evidence_cache_path=evidence_file,
                max_support_turns=6,
                overwrite_cache=False,
            )
            dataloader = HiDialogDataloaderStructuredEvidence(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                relation_num=36,
                max_length=512,
                dataset_name="DialogRE",
            )

            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in dataloader:
                    loss, logits, _ = model(
                        input_ids=batch["input_ids"].to(device),
                        token_type_ids=batch["segment_ids"].to(device),
                        attention_mask=batch["input_masks"].to(device),
                        speaker_ids=batch["speaker_ids"].to(device),
                        graphs=batch["graphs"],
                        labels=batch["label_ids"].to(device),
                        output_attentions=False,
                        cls_indices=batch["cls_indices"].to(device),
                        hidialog_mask=batch["hidialog_masks"].to(device),
                        turn_ids=batch["turn_ids"].to(device),
                        evidence_turn_mask=batch["evidence_turn_mask"].to(device),
                        evidence_turn_scores=batch["evidence_turn_scores"].to(device),
                        relation_prior=batch["relation_prior"].to(device),
                        evidence_global_confidence=batch["evidence_global_confidence"].to(device),
                        evidence_valid=batch["evidence_valid"].to(device),
                    )
                    all_logits.extend(list(logits.detach().cpu().numpy()))
                    all_labels.extend(list(batch["label_ids"].cpu().numpy()))

            f1_val, t2 = f1_eval(all_logits, all_labels)
            results[split] = {"f1": f1_val, "T2": t2}

        return results

    logger.warning("evaluate_model: insufficient arguments provided")
    return {"dev": {"f1": 0.0, "T2": 0.0}, "test": {"f1": 0.0, "T2": 0.0}}