
import json
import logging
import os
import pickle
import random
import re
from collections import defaultdict
from itertools import permutations

import numpy as np
import torch
from torch.utils.data import Dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def load_relation_schema(data_dir):
    schema_path = os.path.join(data_dir, "dialogre_rel2id.json")
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"label2id": {}, "id2label": {}}


def normalize_relation_label(label, label_keys):
    if not label:
        return None
    label = str(label).strip()
    if label in label_keys:
        return label
    for key in label_keys:
        if label.lower() == key.lower():
            return key
    return None


class InputExample:

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None,
                 dialog_id=None, pair_id=None, sample_idx=None, dialog_turns=None,
                 raw_relations=None, raw_relation_ids=None, raw_head=None, raw_tail=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.dialog_id = dialog_id
        self.pair_id = pair_id
        self.sample_idx = sample_idx
        self.dialog_turns = dialog_turns or []
        self.raw_relations = raw_relations or []
        self.raw_relation_ids = raw_relation_ids or []
        self.raw_head = raw_head if raw_head is not None else text_b
        self.raw_tail = raw_tail if raw_tail is not None else text_c


class DialogREProcessor:

    def __init__(self, src_file, n_class, dataset="DialogRE", conversational_f1=False):
        self.src_file = src_file
        self.dataset = dataset
        self.n_class = n_class
        self.conversational_f1 = conversational_f1
        self.extra_info = "MuTual" in str(src_file) or "DDRel" in str(src_file)
        random.seed(42)

    @staticmethod
    def _is_speaker(text):
        text = text.split()
        return len(text) == 2 and text[0] == "speaker" and text[1].isdigit()

    def _rename_arguments(self, dialogue_text, head, tail):
        unused = ["[unused1]", "[unused2]"]
        candidates = []

        if self._is_speaker(head):
            candidates.append(head)
        else:
            candidates.append(None)

        if head != tail and self._is_speaker(tail):
            candidates.append(tail)
        else:
            candidates.append(None)

        for idx, candidate in enumerate(candidates):
            if candidate is None:
                continue
            dialogue_text = dialogue_text.replace(candidate + ":", unused[idx] + " :")
            if head == candidate:
                head = unused[idx]
            if tail == candidate:
                tail = unused[idx]

        return dialogue_text, head, tail

    def _build_label_vector(self, pair):
        if self.dataset in ["DDRel"]:
            return pair["rid"][0] - 1
        label = []
        for relation_idx in range(self.n_class):
            label.append(1 if relation_idx + 1 in pair["rid"] else 0)
        return label

    def _load_split(self, split):
        with open(os.path.join(self.src_file, f"{split}.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def _create_examples_for_split(self, split):
        data = self._load_split(split)
        examples = []
        pair_sample_idx = 0

        for dialog_id, dialog in enumerate(data):
            turns, pairs = dialog
            for pair_id, pair in enumerate(pairs):
                label = self._build_label_vector(pair)

                if self.conversational_f1:
                    prefix_range = range(1, len(turns) + 1)
                else:
                    prefix_range = [len(turns)]

                for prefix_turn_count in prefix_range:
                    renamed_dialog, head, tail = self._rename_arguments(
                        "[TURNEND]\n".join(turns[:prefix_turn_count]).lower(),
                        pair["x"].lower(),
                        pair["y"].lower(),
                    )

                    example_kwargs = {
                        "guid": f"{split}-{len(examples)}",
                        "text_a": renamed_dialog,
                        "text_b": head,
                        "text_c": tail,
                        "label": label,
                        "dialog_id": dialog_id,
                        "pair_id": pair_id,
                        "sample_idx": pair_sample_idx,
                        "dialog_turns": turns[:prefix_turn_count],
                        "raw_relations": pair.get("r", []),
                        "raw_relation_ids": pair.get("rid", []),
                        "raw_head": pair["x"].lower(),
                        "raw_tail": pair["y"].lower(),
                    }

                    examples.append(InputExample(**example_kwargs))

                pair_sample_idx += 1

        return examples

    def get_train_examples(self):
        return self._create_examples_for_split("train")

    def get_dev_examples(self):
        return self._create_examples_for_split("dev")

    def get_test_examples(self):
        return self._create_examples_for_split("test")


class Feature:

    def __init__(self, input_ids, segment_ids, input_mask, speaker_ids, label_id,
                 mention_ids, CLS_indices, transcript_id, answer_id, hidialog_mask,
                 mentioned_h=None, mentioned_t=None):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.speaker_ids = speaker_ids
        self.label_id = label_id
        self.mention_ids = mention_ids
        self.CLS_indices = CLS_indices
        self.transcript_id = transcript_id
        self.answer_id = answer_id
        self.hidialog_mask = hidialog_mask
        self.mentioned_h = mentioned_h if mentioned_h is not None else []
        self.mentioned_t = mentioned_t if mentioned_t is not None else []


def convert_examples_to_features(examples, max_seq_length, tokenizer, dataset="DialogRE"):
    features = []

    for example in examples:
        dialogue = example.text_a
        entity1 = example.text_b
        entity2 = example.text_c

        dialogue_tokens = tokenizer.tokenize(dialogue)
        entity1_tokens = tokenizer.tokenize(entity1)
        entity2_tokens = tokenizer.tokenize(entity2)

        tokens = ["[CLS]"] + dialogue_tokens + ["[SEP]"] + entity1_tokens + ["[SEP]"] + entity2_tokens + ["[SEP]"]

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(tokens)

        sep_idx = input_ids.index(tokenizer.sep_token_id)
        segment_ids = [0] * (sep_idx + 1) + [1] * (len(tokens) - sep_idx - 1)

        input_mask = [1] * len(tokens)
        padding_length = max_seq_length - len(input_ids)

        input_ids += [0] * padding_length
        segment_ids += [0] * padding_length
        input_mask += [0] * padding_length

        features.append([Feature(
            input_ids=input_ids,
            segment_ids=segment_ids,
            input_mask=input_mask,
            speaker_ids=[0] * max_seq_length,
            label_id=example.label,
            mention_ids=[0] * max_seq_length,
            CLS_indices=[False] * max_seq_length,
            transcript_id=0,
            answer_id=0,
            hidialog_mask=[[0] * max_seq_length for _ in range(max_seq_length)]
        )])

    return features


def convert_examples_to_features_roberta(examples, max_seq_length, tokenizer, dataset="DialogRE"):
    features = []
    cls_token_id = getattr(tokenizer, "cls_token_id", 0)
    eos_token_id = getattr(tokenizer, "eos_token_id", 2)
    pad_token_id = getattr(tokenizer, "pad_token_id", 1)

    for example in examples:
        dialogue = example.text_a
        entity1 = example.text_b
        entity2 = example.text_c

        turns = [t.strip() for t in re.split(r"\[TURNEND\]", dialogue, flags=re.IGNORECASE) if t.strip()]
        speaker_id_map = {}
        turn_segments = []

        for turn in turns:
            if ":" in turn:
                speaker, content = turn.split(":", 1)
                speaker = speaker.strip()
                content = content.strip()
            else:
                speaker = "speaker 0"
                content = turn

            speaker_num = int(speaker.split()[-1]) if speaker.split()[-1].isdigit() else 0
            if speaker_num not in speaker_id_map:
                speaker_id_map[speaker_num] = len(speaker_id_map) + 1
            sid = speaker_id_map[speaker_num]

            speaker_tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(speaker + " "))
            content_tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(content))
            turn_segments.append((speaker_tokens_ids, content_tokens_ids, sid))

        entity1_tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + entity1))
        entity2_tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + entity2))

        budget = max_seq_length - 1 - len(entity1_tokens_ids) - len(entity2_tokens_ids) - 1
        kept_segments = []
        used = 0
        for seg in turn_segments:
            if used + len(seg[0]) + len(seg[1]) > budget:
                break
            kept_segments.append(seg)
            used += len(seg[0]) + len(seg[1])

        all_tokens = [cls_token_id]
        mention_ids = [0]
        speaker_ids = [0]
        cls_positions = [0]

        mention_counter = 1
        for speaker_tokens_ids, content_tokens_ids, sid in kept_segments:
            cls_positions.append(len(all_tokens))
            all_tokens.extend(speaker_tokens_ids)
            mention_ids.extend([mention_counter] * len(speaker_tokens_ids))
            speaker_ids.extend([sid] * len(speaker_tokens_ids))
            all_tokens.extend(content_tokens_ids)
            mention_ids.extend([mention_counter] * len(content_tokens_ids))
            speaker_ids.extend([sid] * len(content_tokens_ids))
            mention_counter += 1

        cls_positions.append(len(all_tokens))
        all_tokens.extend(entity1_tokens_ids)
        mention_ids.extend([mention_counter] * len(entity1_tokens_ids))
        speaker_ids.extend([0] * len(entity1_tokens_ids))
        mention_counter += 1

        cls_positions.append(len(all_tokens))
        all_tokens.extend(entity2_tokens_ids)
        mention_ids.extend([mention_counter] * len(entity2_tokens_ids))
        speaker_ids.extend([0] * len(entity2_tokens_ids))

        all_tokens.append(eos_token_id)
        mention_ids.append(mention_ids[-1] if mention_ids else 0)
        speaker_ids.append(0)

        if len(all_tokens) > max_seq_length:
            all_tokens = all_tokens[:max_seq_length]
            mention_ids = mention_ids[:max_seq_length]
            speaker_ids = speaker_ids[:max_seq_length]
            cls_positions = [p for p in cls_positions if p < max_seq_length]

        padding_length = max_seq_length - len(all_tokens)
        all_tokens += [pad_token_id] * padding_length
        mention_ids += [0] * padding_length
        speaker_ids += [0] * padding_length

        CLS_indices = [False] * max_seq_length
        for pos in cls_positions:
            CLS_indices[pos] = True

        unique_mids = sorted(set(m for m in mention_ids if m > 0))
        if len(unique_mids) >= 3:
            head_mid = unique_mids[-2]
            tail_mid = unique_mids[-1]
        else:
            head_mid = unique_mids[-1] if unique_mids else 0
            tail_mid = 0
        mentioned_h = [head_mid]
        mentioned_t = [tail_mid]

        hidialog_mask = np.zeros((max_seq_length, max_seq_length), dtype=np.float32)
        seq_len = max_seq_length - padding_length
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if mention_ids[i] == mention_ids[j] or speaker_ids[i] == speaker_ids[j]:
                    hidialog_mask[i, j] = 1.0
                    hidialog_mask[j, i] = 1.0

        features.append([Feature(
            input_ids=all_tokens,
            segment_ids=[0] * max_seq_length,
            input_mask=[1 if t != pad_token_id else 0 for t in all_tokens],
            speaker_ids=speaker_ids,
            label_id=example.label,
            mention_ids=mention_ids,
            CLS_indices=CLS_indices,
            transcript_id=0,
            answer_id=0,
            hidialog_mask=hidialog_mask,
            mentioned_h=mentioned_h,
            mentioned_t=mentioned_t,
        )])

    return features
