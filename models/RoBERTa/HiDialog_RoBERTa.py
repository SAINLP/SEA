
import math
from collections import defaultdict
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

import dgl
import dgl.nn.pytorch as dglnn
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class RobertaEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.speaker_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.padding_idx = config.pad_token_id
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, speaker_ids=None):
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if speaker_ids is not None:
            speaker_embeddings = self.speaker_embeddings(speaker_ids)
            embeddings = inputs_embeds + token_type_embeddings + speaker_embeddings
        else:
            embeddings = inputs_embeds + token_type_embeddings

        if hasattr(self, 'position_embedding_type') and self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long() + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, 
                                     dtype=torch.long, device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape)


class RobertaSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"Hidden size {config.hidden_size} not divisible by num_attention_heads {config.num_attention_heads}")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer, attention_probs) if output_attentions else (context_layer,)


class RobertaSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class RobertaIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RobertaOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return (layer_output,) + self_attention_outputs[1:]


class RobertaEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return (hidden_states, all_hidden_states, all_self_attentions)


class RobertaPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return self.activation(pooled_output)


class RobertaPreTrainedModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        from transformers import AutoModel
        return AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.weight.size(1))
        min_size = min(old_embeddings.weight.size(0), new_num_tokens)
        new_embeddings.weight.data[:min_size] = old_embeddings.weight.data[:min_size]
        self.set_input_embeddings(new_embeddings)
        return new_embeddings

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RobertaModel(RobertaPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                speaker_ids=None, head_mask=None, inputs_embeds=None, output_attentions=False, 
                output_hidden_states=False, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids, inputs_embeds, speaker_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask, output_attentions, output_hidden_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output, encoder_outputs[1], encoder_outputs[2])

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(f"Unsupported attention mask shape: {attention_mask.shape}")

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers):
        if head_mask is None:
            return [None] * num_hidden_layers
        return head_mask


class GTConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.1)

    def forward(self, A):
        A = A.to(self.weight.device)
        A = torch.sum(A * F.softmax(self.weight, dim=1), dim=1)
        return A


class GTLayer(nn.Module):

    def __init__(self, in_channels, out_channels, first=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first

        if self.first:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):
        if self.first:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a, b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach().cpu(),
                 (F.softmax(self.conv2.weight, dim=1)).detach().cpu()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_, a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach().cpu()]
        return H, W


class GTN(nn.Module):

    def __init__(self, etypes, num_channels, w_in, w_out, num_layers, norm=True, num_class=None):
        super().__init__()
        self.etypes = etypes
        self.num_edge = len(etypes) + 1
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(self.num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(self.num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)

        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))

        if num_class is not None:
            self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)
            self.linear2 = nn.Linear(self.w_out, self.num_class)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self, X, H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(), X)

    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor)).to(H.device)
        else:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.FloatTensor)).to(H.device) + torch.eye(H.shape[0]).type(torch.FloatTensor).to(H.device)

        deg = torch.sum(H, dim=1).to(H.device)
        deg_inv = (deg + 1e-8).pow(-1)
        deg_inv[deg == 0] = 0
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv * torch.eye(H.shape[0]).type(torch.FloatTensor).to(H.device)
        H = torch.mm(deg_inv, H)
        return H.t()

    def adj(self, graph):
        num_nodes = graph.number_of_nodes('node')
        for k, etype in enumerate(self.etypes):
            A_k = graph.adj(etype=etype)
            if hasattr(A_k, "to_dense"):
                A_k = A_k.to_dense()
            A_k = A_k.to(dtype=torch.float32)
            if k == 0:
                A = A_k.unsqueeze(-1)
            else:
                A = torch.cat([A, A_k.unsqueeze(-1)], dim=-1)

        A = torch.cat([A, torch.eye(num_nodes, device=A.device, dtype=torch.float32).unsqueeze(-1)], dim=-1)
        return A

    def forward(self, graph, X, target_x=None, target=None, output_gtn_adjacency=False):
        A = self.adj(graph)
        A = A.unsqueeze(0).permute(0, 3, 1, 2)
        Ws = []

        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                for j in range(self.num_channels):
                    if j == 0:
                        H_ = self.norm(H[j], add=True).unsqueeze(0)
                    else:
                        H_ = torch.cat((H_, self.norm(H[j], add=True).unsqueeze(0)), dim=0)
                H = H_
                H, W = self.layers[i](A, H)
            Ws.append(W)

        for i in range(self.num_channels):
            if i == 0:
                X_ = F.relu(self.gcn_conv(X, H[i]))
            else:
                X_tmp = F.relu(self.gcn_conv(X, H[i]))
                X_ = torch.cat((X_, X_tmp), dim=1)

        if target is None:
            Hs = torch.stack([self.norm(H_, add=True) for H_ in H]).detach().cpu() if output_gtn_adjacency else None
            return X_, Ws, Hs
        else:
            X_ = self.linear1(X_)
            X_ = F.relu(X_)
            y = self.linear2(X_[target_x])
            return y, Ws


class HiDialog_RoBERTa(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_labels, gtn_out=1024, activation='relu', 
                 gtn_num_channels=3, gtn_num_layers=2, dataset='DialogRE'):
        super().__init__(config)
        self.num_labels = num_labels
        self.dataset = dataset
        self.node_dim = gtn_out * gtn_num_channels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.etypes = ["dialog", "sequence", "entity", "speaker"]
        self.gtn = GTN(self.etypes, gtn_num_channels, config.hidden_size, gtn_out, gtn_num_layers)

        self.classifier = nn.Linear(self.node_dim * 3, self.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, speaker_ids=None,
                graphs=None, hidialog_mask=None, cls_indices=None, labels=None, output_attentions=False):

        outputs = self.roberta(input_ids, attention_mask=hidialog_mask, speaker_ids=speaker_ids)
        sequence_outputs = outputs[0]

        graph_outputs = []
        for batch_idx, graph in enumerate(graphs):
            sequence_output = sequence_outputs[batch_idx]
            mask = cls_indices[batch_idx].unsqueeze(-1).expand(-1, sequence_output.shape[-1])
            features_graph = torch.masked_select(sequence_output, mask).view(-1, sequence_output.shape[-1])

            graph_features, _, _ = self.gtn(graph, features_graph, output_gtn_adjacency=output_attentions)
            graph_outputs.append(graph_features)

        pooled_output = self.dropout(torch.stack(graph_outputs))
        logits = self.classifier(pooled_output).view(-1, self.num_labels)

        loss = None
        if labels is not None:
            if self.dataset not in ["MRDA", "MuTual"]:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.view(-1, self.num_labels))
            elif self.dataset in ["MRDA"]:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            elif self.dataset in ["MuTual"]:
                logits = logits.view(-1, 4)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, torch.argmax(labels.view(-1, 4), dim=1))

        if labels is not None:
            return loss, logits
        return logits


class SlotGATConv(nn.Module):

    def __init__(self, edge_dim, num_etypes, in_feats, out_feats, num_heads,
                 feat_drop=0.1, attn_drop=0.1, negative_slope=0.2, residual=False,
                 activation=None, alpha=0.1):
        super().__init__()

        self._edge_dim = edge_dim
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.edge_emb = nn.Embedding(num_etypes, edge_dim)
        self.fc = nn.Linear(in_feats, out_feats * num_heads)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_dim)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            self.res_fc = nn.Linear(in_feats, out_feats * num_heads) if in_feats != out_feats else None
        else:
            self.res_fc = None

        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.edge_emb.weight, gain=gain)

    def forward(self, graph, feat, e_feat=None):
        h_src = self.feat_drop(feat)
        h_proj = self.fc(h_src).view(-1, self._num_heads, self._out_feats)

        el = (h_proj * self.attn_l).sum(dim=-1).unsqueeze(-1)

        all_edges = []
        all_attention = []

        for etype in graph.etypes:
            try:
                src, dst = graph.edges(etype=etype)
                if len(src) > 0:
                    e_type = el[src] + el[dst]
                    e_type = self.leaky_relu(e_type)

                    subgraph = dgl.graph((src, dst), num_nodes=graph.number_of_nodes('node'))
                    subgraph.edata['e'] = e_type
                    a_type = self.attn_drop(edge_softmax(subgraph, subgraph.edata['e']))

                    all_edges.append((src, dst))
                    all_attention.append(a_type)
            except:
                continue

        if all_edges:
            rst = torch.zeros_like(h_proj)
            for (src, dst), a_type in zip(all_edges, all_attention):
                messages = h_proj[src] * a_type
                for j, dst_node in enumerate(dst):
                    rst[dst_node] += messages[j]

            if self.res_fc is not None:
                rst = rst + self.res_fc(h_src).view(-1, self._num_heads, self._out_feats)

            if self.activation:
                rst = self.activation(rst)

            return rst.flatten(1), all_attention[0] if all_attention else None
        else:
            return feat, None


class SlotGATModule(nn.Module):

    def __init__(self, edge_dim, num_etypes, in_dims, num_hidden, num_classes, num_layers,
                 heads, feat_drop=0.1, attn_drop=0.1, negative_slope=0.2, residual=True, alpha=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        self.input_proj = nn.Linear(in_dims, num_hidden)

        self.gat_layers = nn.ModuleList()
        for l in range(num_layers):
            in_feats = num_hidden if l == 0 else num_hidden * heads[l-1]
            self.gat_layers.append(SlotGATConv(
                edge_dim, num_etypes, in_feats, num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual if l > 0 else False,
                nn.ReLU() if l < num_layers - 1 else None, alpha
            ))

        self.final_proj = nn.Linear(num_hidden, num_classes)

    def forward(self, graph, features, output_attention=False):
        h = self.input_proj(features)

        for l in range(self.num_layers):
            h, attn = self.gat_layers[l](graph, h)

        logits = self.final_proj(h)
        return logits, attn if output_attention else None
