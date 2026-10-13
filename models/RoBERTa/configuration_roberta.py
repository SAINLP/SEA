
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class RobertaConfig:

    model_type = "roberta"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = cls(**config_dict)
        for key, value in config_dict.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        return config

    def to_dict(self):
        output = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                output[key] = value
        return output
