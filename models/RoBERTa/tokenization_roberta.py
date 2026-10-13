
import json
import logging
import os
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

SPECIAL_TOKENS_ATTRIBUTES = [
    "bos_token", "eos_token", "unk_token", "sep_token",
    "pad_token", "cls_token", "mask_token",
    "additional_special_tokens",
]


class RobertaTokenizer:

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        errors: str = "replace",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        cls_token: str = "<s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        add_prefix_space: bool = False,
        trim_offsets: bool = True,
        **kwargs,
    ):
        try:
            from tokenizers import Tokenizer
            from tokenizers.implementations import ByteLevelBPETokenizer
        except ImportError as e:
            raise ImportError(
                "The `tokenizers` library is required for RobertaTokenizer. "
                "Install it with: pip install tokenizers==0.19.1"
            ) from e

        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.tokenizer_file = tokenizer_file
        self.errors = errors
        self.add_prefix_space = add_prefix_space

        self._bos_token = bos_token
        self._eos_token = eos_token
        self._sep_token = sep_token
        self._cls_token = cls_token
        self._unk_token = unk_token
        self._pad_token = pad_token
        self._mask_token = mask_token
        self._additional_special_tokens: List[str] = []

        if tokenizer_file and os.path.exists(tokenizer_file):
            self._tokenizer = Tokenizer.from_file(tokenizer_file)
            if add_prefix_space:
                logger.warning(
                    "tokenizer_file already contains a ByteLevel pre-tokenizer config, "
                    "ignoring add_prefix_space=%s",
                    add_prefix_space,
                )
        elif vocab_file and merges_file:
            self._tokenizer = ByteLevelBPETokenizer(
                vocab=vocab_file,
                merges=merges_file,
                add_prefix_space=add_prefix_space,
                trim_offsets=trim_offsets,
            )
        else:
            raise ValueError(
                "Either tokenizer_file (tokenizer.json) or vocab_file + merges_file must be provided"
            )

        self._vocab: Dict[str, int] = self._tokenizer.get_vocab()
        self._ids_to_tokens: Dict[int, str] = {v: k for k, v in self._vocab.items()}

        self._bos_token_id = self._vocab.get(self._bos_token)
        self._eos_token_id = self._vocab.get(self._eos_token)
        self._sep_token_id = self._vocab.get(self._sep_token)
        self._cls_token_id = self._vocab.get(self._cls_token)
        self._unk_token_id = self._vocab.get(self._unk_token)
        self._pad_token_id = self._vocab.get(self._pad_token)
        self._mask_token_id = self._vocab.get(self._mask_token)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos_token_id

    @property
    def sep_token_id(self) -> Optional[int]:
        return self._sep_token_id

    @property
    def cls_token_id(self) -> Optional[int]:
        return self._cls_token_id

    @property
    def unk_token_id(self) -> Optional[int]:
        return self._unk_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._pad_token_id

    @property
    def mask_token_id(self) -> Optional[int]:
        return self._mask_token_id

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def sep_token(self) -> str:
        return self._sep_token

    @property
    def cls_token(self) -> str:
        return self._cls_token

    @property
    def unk_token(self) -> str:
        return self._unk_token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def mask_token(self) -> str:
        return self._mask_token

    @property
    def additional_special_tokens(self) -> List[str]:
        return list(self._additional_special_tokens)

    @property
    def all_special_tokens(self) -> List[str]:
        return [
            self._bos_token, self._eos_token, self._unk_token, self._sep_token,
            self._pad_token, self._cls_token, self._mask_token,
        ] + list(self._additional_special_tokens)

    @property
    def all_special_ids(self) -> List[int]:
        ids = [
            self._bos_token_id, self._eos_token_id, self._unk_token_id,
            self._sep_token_id, self._pad_token_id, self._cls_token_id,
            self._mask_token_id,
        ]
        return [i for i in ids if i is not None] + [
            self._vocab[t] for t in self._additional_special_tokens if t in self._vocab
        ]

    def tokenize(self, text: str, **kwargs) -> List[str]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text)}")
        return self._tokenizer.encode(text, add_special_tokens=False).tokens

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> List[int]:
        if isinstance(tokens, str):
            return self._tokenizer.encode(tokens, add_special_tokens=False).ids
        unk = self._unk_token_id
        return [self._vocab.get(t, unk) for t in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._ids_to_tokens.get(ids, self._unk_token)
        special_ids = set(self.all_special_ids) if skip_special_tokens else set()
        return [
            self._ids_to_tokens.get(i, self._unk_token)
            for i in ids if i not in special_ids
        ]

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return [self._cls_token_id] + token_ids_0 + [self._sep_token_id]
        return (
            [self._cls_token_id] + token_ids_0 + [self._sep_token_id]
            + token_ids_1 + [self._sep_token_id]
        )

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2 if not pair else 3

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        return [0] * self.num_special_tokens_to_add(token_ids_1 is not None)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if add_special_tokens:
            return [self._cls_token_id] + ids + [self._sep_token_id]
        return list(ids)

    def decode(self, ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True) -> str:
        if skip_special_tokens:
            special_ids = set(self.all_special_ids)
            ids = [i for i in ids if i not in special_ids]
        return self._tokenizer.decode(ids)

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> Dict[str, List[int]]:
        if isinstance(text, list):
            batch = []
            for item in text:
                if text_pair is not None:
                    item_pair = text_pair[0] if isinstance(text_pair, list) else text_pair
                    batch.append(self.encode_pair(item, item_pair, add_special_tokens))
                else:
                    batch.append(self.encode(item, add_special_tokens))
            return self._pad_batch(batch)
        if text_pair is not None:
            ids = self.encode_pair(text, text_pair, add_special_tokens)
        else:
            ids = self.encode(text, add_special_tokens)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def encode_pair(self, text: str, text_pair: str, add_special_tokens: bool = True) -> List[int]:
        ids_0 = self._tokenizer.encode(text, add_special_tokens=False).ids
        ids_1 = self._tokenizer.encode(text_pair, add_special_tokens=False).ids
        if add_special_tokens:
            return self.build_inputs_with_special_tokens(ids_0, ids_1)
        return ids_0 + ids_1

    def _pad_batch(self, batch: List[List[int]]) -> Dict[str, List[int]]:
        max_len = max(len(x) for x in batch)
        input_ids = [x + [self._pad_token_id] * (max_len - len(x)) for x in batch]
        attention_mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in batch]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, List[str]]]) -> int:
        added = 0
        for key, value in special_tokens_dict.items():
            if key not in SPECIAL_TOKENS_ATTRIBUTES:
                continue
            if key == "additional_special_tokens":
                for token in value:
                    if token not in self._vocab:
                        self._tokenizer.add_special_tokens([token])
                        added += 1
                    self._additional_special_tokens.append(token)
            else:
                setattr(self, "_" + key, value)
        if added:
            self._vocab = self._tokenizer.get_vocab()
            self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
            self._bos_token_id = self._vocab.get(self._bos_token)
            self._eos_token_id = self._vocab.get(self._eos_token)
            self._sep_token_id = self._vocab.get(self._sep_token)
            self._cls_token_id = self._vocab.get(self._cls_token)
            self._unk_token_id = self._vocab.get(self._unk_token)
            self._pad_token_id = self._vocab.get(self._pad_token)
            self._mask_token_id = self._vocab.get(self._mask_token)
        return added

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> List[str]:
        os.makedirs(save_directory, exist_ok=True)
        vocab_name = f"{filename_prefix}vocab.json" if filename_prefix else "vocab.json"
        merges_name = f"{filename_prefix}merges.txt" if filename_prefix else "merges.txt"
        vocab_path = os.path.join(save_directory, vocab_name)
        merges_path = os.path.join(save_directory, merges_name)

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False)

        if self.merges_file and os.path.exists(self.merges_file):
            with open(self.merges_file, "r", encoding="utf-8") as src:
                with open(merges_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
        return [vocab_path, merges_path]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(vocab_size={self.vocab_size}, "
            f"vocab_file={self.vocab_file!r}, merges_file={self.merges_file!r}, "
            f"tokenizer_file={self.tokenizer_file!r})"
        )
