
import json
import logging
import math
import os
import pickle
import random
from collections import defaultdict
from itertools import permutations

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from data import convert_examples_to_features, convert_examples_to_features_roberta, get_cuda
from dialogre.constants import load_relation_schema, normalize_relation_label

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

FORMAT_VERSION = "structured_evidence_v3"


def _load_evidence_cache(evidence_cache_path):
    if not evidence_cache_path or not os.path.exists(evidence_cache_path):
        return {}
    with open(evidence_cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_sample_key(split_name, example):
    return (
        f"{split_name}||{example.dialog_id}||{example.raw_head}||"
        f"{example.raw_tail}||{example.pair_id}"
    )


def _normalize_support_turns(support_turns, num_dialog_turns, max_support_turns):
    if not isinstance(support_turns, list):
        return []

    cleaned = []
    for value in support_turns:
        try:
            value = int(value)
        except Exception:
            continue
        if 1 <= value <= num_dialog_turns:
            cleaned.append(value)

    cleaned = sorted(set(cleaned))
    return cleaned[:max_support_turns]


def _normalize_turn_confidences(support_turns, evidence_record, global_confidence):
    confidence_values = evidence_record.get("turn_confidences", [])
    scores = {}

    if isinstance(confidence_values, list) and len(confidence_values) == len(support_turns):
        for turn_id, confidence in zip(support_turns, confidence_values):
            try:
                scores[turn_id] = max(float(confidence), 0.0)
            except Exception:
                scores[turn_id] = 0.0
    else:
        for rank, turn_id in enumerate(support_turns):
            scores[turn_id] = max(global_confidence / float(rank + 1), 0.0)

    return scores


def _relation_distribution(candidate_relations, label2id, num_labels):
    dist = np.zeros(num_labels, dtype=np.float32)
    if not isinstance(candidate_relations, list):
        return dist

    for item in candidate_relations:
        if isinstance(item, dict):
            label = normalize_relation_label(str(item.get("label", "")).strip(), label2id.keys())
            if label is None:
                continue
            try:
                score = max(float(item.get("score", 0.0)), 0.0)
            except Exception:
                score = 0.0
        elif isinstance(item, str):
            label = normalize_relation_label(item.strip(), label2id.keys())
            if label is None:
                continue
            score = 1.0
        else:
            continue
        dist[label2id[label]] += score

    total = dist.sum()
    if total > 0:
        dist = dist / total
    return dist


class HiDialogDatasetStructuredEvidence(IterableDataset):

    def __init__(
        self,
        src_file,
        save_file,
        max_seq_length,
        tokenizer,
        n_class,
        encoder_type,
        dataset="DialogRE",
        split_name=None,
        evidence_cache_path=None,
        max_support_turns=6,
        overwrite_cache=False,
    ):
        super().__init__()
        self.data = None
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.relation_schema = load_relation_schema(src_file) if dataset in ("DialogRE", "MELD") else None
        evidence_cache = _load_evidence_cache(evidence_cache_path)

        logger.info("Reading data from %s", src_file)

        if os.path.exists(save_file) and not overwrite_cache:
            with open(save_file, "rb") as fr:
                info = pickle.load(fr)
            if info.get("format_version") == FORMAT_VERSION:
                self.data = info["data"]
                logger.info("Loaded preprocessed data from %s", save_file)
                return

        from data import DialogREProcessor
        processor = DialogREProcessor(
            src_file=src_file,
            n_class=n_class,
            dataset=dataset,
            conversational_f1=split_name in {"devc", "testc"},
        )

        if split_name == "train":
            examples = processor.get_train_examples()
        elif split_name in {"dev", "devc"}:
            examples = processor.get_dev_examples()
        elif split_name in {"test", "testc"}:
            examples = processor.get_test_examples()
        else:
            raise ValueError(f"Unsupported split_name: {split_name}")

        if encoder_type in ["BERT", "DeBERTa", "Electra"]:
            features = convert_examples_to_features(examples, max_seq_length, tokenizer, dataset=dataset)
        else:
            features = convert_examples_to_features_roberta(examples, max_seq_length, tokenizer, dataset=dataset)

        self.data = []
        for idx, feature_list in enumerate(features):
            feature = feature_list[0]
            example = examples[idx]

            entity_edges_infor = {"h": set(getattr(feature, 'mentioned_h', []) or []),
                                   "t": set(getattr(feature, 'mentioned_t', []) or [])}
            speaker_infor = self.make_speaker_infor(feature.speaker_ids, feature.mention_ids)
            max_mid = max(feature.mention_ids)
            if max_mid < 3:
                max_mid = 3
            turn_node_num = max_mid - 2
            head_mention_id = max_mid - 1
            tail_mention_id = max_mid
            entity_edges_infor["h"].add(head_mention_id)
            entity_edges_infor["t"].add(tail_mention_id)
            graph, used_mention = self.create_graph(
                speaker_infor,
                turn_node_num,
                entity_edges_infor,
                head_mention_id,
                tail_mention_id,
            )

            split_key = "dev" if split_name == "devc" else "test" if split_name == "testc" else split_name
            sample_key = _make_sample_key(split_key, example)
            evidence_record = evidence_cache.get(sample_key, {})

            support_turns = _normalize_support_turns(
                evidence_record.get("support_turns", []),
                len(example.dialog_turns),
                max_support_turns,
            )

            relation_prior = (
                _relation_distribution(
                    evidence_record.get("candidate_relations", []),
                    self.relation_schema["label2id"],
                    n_class,
                )
                if self.relation_schema is not None
                else np.zeros(n_class, dtype=np.float32)
            )

            if "global_confidence" in evidence_record:
                try:
                    global_confidence = max(float(evidence_record["global_confidence"]), 0.0)
                except Exception:
                    global_confidence = 0.0
            elif relation_prior.sum() > 0:
                global_confidence = float(relation_prior.max())
            elif support_turns:
                global_confidence = 1.0
            else:
                global_confidence = 0.0

            turn_count = turn_node_num + 1
            turn_ids = np.arange(turn_count, dtype=np.int64)
            evidence_turn_mask = np.zeros(turn_count, dtype=np.float32)
            evidence_turn_scores = np.zeros(turn_count, dtype=np.float32)
            turn_conf_map = _normalize_turn_confidences(support_turns, evidence_record, global_confidence)

            for turn_id in support_turns:
                if 0 <= turn_id < turn_count:
                    evidence_turn_mask[turn_id] = 1.0
                    evidence_turn_scores[turn_id] = turn_conf_map.get(turn_id, global_confidence)

            if evidence_turn_scores.sum() > 0:
                evidence_turn_scores = evidence_turn_scores / evidence_turn_scores.sum()

            evidence_valid = float(
                bool(support_turns) or bool(evidence_record.get("candidate_relations", []))
            )

            if dataset in ["DDRel"]:
                label_ids_out = feature.label_id
            else:
                label_ids_out = np.array(feature.label_id)

            self.data.append({
                    "input_ids": np.array(feature.input_ids, dtype=np.int64),
                    "segment_ids": np.array(feature.segment_ids),
                    "input_mask": np.array(feature.input_mask),
                    "speaker_ids": np.array(feature.speaker_ids),
                    "label_ids": label_ids_out,
                    "mention_id": np.array(feature.mention_ids),
                    "graph": graph,
                    "cls_indices": np.array(feature.CLS_indices, dtype=bool),
                    "transcript_ids": np.array(feature.transcript_id),
                    "answer_ids": np.array(feature.answer_id),
                    "hidialog_mask": np.array(feature.hidialog_mask, dtype=np.uint8),
                    "turn_ids": turn_ids,
                    "turn_count": turn_count,
                    "evidence_turn_mask": evidence_turn_mask,
                    "evidence_turn_scores": evidence_turn_scores,
                    "relation_prior": relation_prior,
                    "evidence_global_confidence": np.array(global_confidence, dtype=np.float32),
                    "evidence_valid": np.array(evidence_valid, dtype=np.float32),
                    "sample_key": sample_key,
                })

        with open(save_file, "wb") as fw:
            pickle.dump({"format_version": FORMAT_VERSION, "data": self.data}, fw)
        logger.info("Finished reading %s and saved preprocessed data to %s", src_file, save_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    @staticmethod
    def make_speaker_infor(speaker_ids, mention_ids):
        mapping = defaultdict(set)
        for idx in range(1, len(speaker_ids)):
            if speaker_ids[idx] == 0:
                break
            mapping[speaker_ids[idx]].add(mention_ids[idx])
        return {key: list(value) for key, value in mapping.items()}

    @staticmethod
    def create_graph(speaker_infor, turn_node_num, entity_edges_infor, head_mention_id, tail_mention_id):
        d = defaultdict(list)
        used_mention = {0}

        for _, mentions in speaker_infor.items():
            for h, t in permutations(mentions, 2):
                d[("node", "speaker", "node")].append((h, t))
                used_mention.add(h)
                used_mention.add(t)
        if d[("node", "speaker", "node")] == []:
            d[("node", "speaker", "node")].append((1, 0))
            used_mention.add(1)

        for i in range(1, turn_node_num + 1):
            d[("node", "dialog", "node")].append((i, 0))
            d[("node", "dialog", "node")].append((0, i))
            used_mention.add(i)
        if d[("node", "dialog", "node")] == []:
            d[("node", "dialog", "node")].append((1, 0))
            used_mention.add(1)

        for mention in entity_edges_infor["h"]:
            d[("node", "entity", "node")].append((head_mention_id, mention))
            d[("node", "entity", "node")].append((mention, head_mention_id))
            used_mention.add(head_mention_id)
        if entity_edges_infor["h"] == []:
            for i in range(1, turn_node_num + 1):
                d[("node", "entity", "node")].append((head_mention_id, i))
                d[("node", "entity", "node")].append((i, head_mention_id))

        for mention in entity_edges_infor["t"]:
            d[("node", "entity", "node")].append((tail_mention_id, mention))
            d[("node", "entity", "node")].append((mention, tail_mention_id))
            used_mention.add(tail_mention_id)
        if entity_edges_infor["t"] == []:
            for i in range(1, turn_node_num + 1):
                d[("node", "entity", "node")].append((tail_mention_id, i))
                d[("node", "entity", "node")].append((i, tail_mention_id))

        turns = [i for i in range(1, turn_node_num + 1)]
        if len(turns) >= 2:
            for h, t in permutations(turns, 2):
                d[("node", "sequence", "node")].append((h, t))
                used_mention.add(h)
        if d[("node", "sequence", "node")] == []:
            d[("node", "sequence", "node")].append((1, 0))
            used_mention.add(1)

        graph = dgl.heterograph(d)
        return graph, used_mention


class HiDialogDataloaderStructuredEvidence(DataLoader):

    def __init__(self, dataset, batch_size, shuffle=False, relation_num=36, max_length=512, dataset_name="DialogRE"):
        super().__init__(dataset, batch_size=batch_size, worker_init_fn=np.random.seed(42))
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length
        self.relation_num = relation_num
        self.order = list(range(self.length))
        self.dataset_name = dataset_name

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.order)
            self.data = [self.dataset[idx] for idx in self.order]
        else:
            self.data = self.dataset

        batch_num = math.ceil(self.length / self.batch_size)
        self.batches = [
            self.data[idx * self.batch_size : min(self.length, (idx + 1) * self.batch_size)]
            for idx in range(0, batch_num)
        ]

        input_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        segment_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        input_mask = torch.LongTensor(self.batch_size, self.max_length).cpu()
        mention_id = torch.LongTensor(self.batch_size, self.max_length).cpu()
        speaker_id = torch.LongTensor(self.batch_size, self.max_length).cpu()
        label_ids = torch.Tensor(self.batch_size, self.relation_num).cpu()
        cls_indices = torch.Tensor(self.batch_size, self.max_length).bool().cpu()
        hidialog_masks = torch.ByteTensor(self.batch_size, self.max_length, self.max_length).cpu()
        relation_prior = torch.FloatTensor(self.batch_size, self.relation_num).cpu()
        evidence_global_confidence = torch.FloatTensor(self.batch_size).cpu()
        evidence_valid = torch.FloatTensor(self.batch_size).cpu()

        for minibatch in self.batches:
            cur_bsz = len(minibatch)
            max_turn_count = max(example["turn_count"] for example in minibatch)
            turn_ids = torch.LongTensor(self.batch_size, max_turn_count).cpu()
            evidence_turn_mask = torch.FloatTensor(self.batch_size, max_turn_count).cpu()
            evidence_turn_scores = torch.FloatTensor(self.batch_size, max_turn_count).cpu()

            for mapping in [
                input_ids, segment_ids, input_mask, mention_id, label_ids,
                speaker_id, cls_indices, hidialog_masks, relation_prior,
                evidence_global_confidence, evidence_valid,
                turn_ids, evidence_turn_mask, evidence_turn_scores,
            ]:
                mapping.zero_()

            graph_list = []
            sample_keys = []

            for i, example in enumerate(minibatch):
                graph_list.append(example["graph"])
                sample_keys.append(example["sample_key"])

                word_num = example["input_ids"].shape[0]
                input_ids[i, :word_num].copy_(torch.from_numpy(example["input_ids"]))
                segment_ids[i, :word_num].copy_(torch.from_numpy(example["segment_ids"]))
                input_mask[i, :word_num].copy_(torch.from_numpy(example["input_mask"]))
                mention_id[i, :word_num].copy_(torch.from_numpy(example["mention_id"]))
                speaker_id[i, :word_num].copy_(torch.from_numpy(example["speaker_ids"]))
                cls_indices[i, :word_num].copy_(torch.from_numpy(example["cls_indices"]))
                hidialog_masks[i, :word_num, :word_num].copy_(
                    torch.from_numpy(example["hidialog_mask"].astype(np.uint8))
                )
                relation_prior[i].copy_(torch.from_numpy(example["relation_prior"]))
                evidence_global_confidence[i] = float(example["evidence_global_confidence"])
                evidence_valid[i] = float(example["evidence_valid"])

                turn_count = example["turn_count"]
                turn_ids[i, :turn_count].copy_(torch.from_numpy(example["turn_ids"]))
                evidence_turn_mask[i, :turn_count].copy_(torch.from_numpy(example["evidence_turn_mask"]))
                evidence_turn_scores[i, :turn_count].copy_(torch.from_numpy(example["evidence_turn_scores"]))

                relation_num = example["label_ids"].shape[0]
                label_ids[i, :relation_num].copy_(torch.from_numpy(example["label_ids"]))

            context_word_mask = input_mask > 0
            context_word_length = context_word_mask.sum(1)
            batch_max_length = context_word_length.max()

            yield {
                "input_ids": get_cuda(input_ids[:cur_bsz, :batch_max_length].contiguous()),
                "segment_ids": get_cuda(segment_ids[:cur_bsz, :batch_max_length].contiguous()),
                "input_masks": get_cuda(input_mask[:cur_bsz, :batch_max_length].contiguous()),
                "mention_ids": get_cuda(mention_id[:cur_bsz, :batch_max_length].contiguous()),
                "speaker_ids": get_cuda(speaker_id[:cur_bsz, :batch_max_length].contiguous()),
                "label_ids": get_cuda(label_ids[:cur_bsz, : self.relation_num].clone()),
                "graphs": graph_list,
                "cls_indices": get_cuda(cls_indices[:cur_bsz, :batch_max_length].contiguous()),
                "hidialog_masks": get_cuda(hidialog_masks[:cur_bsz, :batch_max_length, :batch_max_length].contiguous()),
                "turn_ids": get_cuda(turn_ids[:cur_bsz, :max_turn_count].contiguous()),
                "evidence_turn_mask": get_cuda(evidence_turn_mask[:cur_bsz, :max_turn_count].contiguous()),
                "evidence_turn_scores": get_cuda(evidence_turn_scores[:cur_bsz, :max_turn_count].contiguous()),
                "relation_prior": get_cuda(relation_prior[:cur_bsz, : self.relation_num].contiguous()),
                "evidence_global_confidence": get_cuda(evidence_global_confidence[:cur_bsz].contiguous()),
                "evidence_valid": get_cuda(evidence_valid[:cur_bsz].contiguous()),
                "sample_keys": sample_keys,
            }
