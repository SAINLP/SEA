
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from .HiDialog_RoBERTa import GTN, RobertaModel, RobertaPreTrainedModel, SlotGATModule


class StructuredEvidenceAdapter(nn.Module):

    def __init__(
        self,
        node_dim,
        base_dim,
        dropout=0.1,
        evidence_bias_scale=0.7,
        use_confidence_gate=True,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.base_dim = base_dim
        self.evidence_bias_scale = evidence_bias_scale
        self.use_confidence_gate = use_confidence_gate

        self.query_proj = nn.Linear(base_dim, node_dim)
        self.key_proj = nn.Linear(node_dim, node_dim)

        self.turn_gate_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
        )

        self.evidence_proj = nn.Linear(node_dim, base_dim)

        self.residual_gate = nn.Sequential(
            nn.Linear(base_dim + node_dim + 1, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        turn_features,
        base_feature,
        evidence_turn_mask,
        evidence_turn_scores,
        global_confidence,
        use_structured_evidence=True,
    ):
        pair_query = torch.tanh(self.query_proj(base_feature))

        base_logits = torch.matmul(self.key_proj(turn_features), pair_query) / math.sqrt(self.node_dim)

        confidence_gate = torch.zeros_like(evidence_turn_scores)

        if use_structured_evidence and evidence_turn_mask.sum() > 0 and evidence_turn_scores.sum() > 0:
            if self.use_confidence_gate:
                gate_input = torch.cat(
                    [
                        turn_features,
                        pair_query.unsqueeze(0).expand_as(turn_features),
                        evidence_turn_scores.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                confidence_gate = torch.sigmoid(self.turn_gate_mlp(gate_input)).squeeze(-1)
            else:
                confidence_gate = torch.ones_like(evidence_turn_scores)

            bias = (
                self.evidence_bias_scale
                * global_confidence.squeeze(0)
                * evidence_turn_mask
                * evidence_turn_scores
                * confidence_gate
            )
            attention_probs = F.softmax(base_logits + bias, dim=-1)
        else:
            attention_probs = F.softmax(base_logits, dim=-1)

        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs / attention_probs.sum().clamp(min=1e-8)

        evidence_summary = torch.sum(attention_probs.unsqueeze(-1) * turn_features, dim=0)

        residual_gate = torch.zeros(1, device=turn_features.device)
        enhanced_feature = base_feature

        if use_structured_evidence and evidence_turn_mask.sum() > 0 and evidence_turn_scores.sum() > 0:
            gate_input = torch.cat([base_feature, evidence_summary, global_confidence.view(1)], dim=-1)
            residual_gate = torch.sigmoid(self.residual_gate(gate_input)).view(1)
            enhanced_feature = base_feature + residual_gate * self.evidence_proj(evidence_summary)

        return enhanced_feature, attention_probs, confidence_gate, residual_gate


class HiDialog_RoBERTa_SEA(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(
        self,
        config,
        num_labels,
        gtn_out=1024,
        activation="relu",
        gtn_num_channels=3,
        gtn_num_layers=2,
        dataset="DialogRE",
        evidence_bias_scale=0.7,
        consistency_loss_weight=0.02,
        relation_prior_weight=0.05,
        use_structured_evidence=True,
        use_confidence_gate=True,
        use_consistency_loss=True,
        use_relation_prior=False,
        use_slotgat=False,
        slotgat_edge_dim=64,
        slotgat_hidden_dim=512,
        slotgat_num_layers=2,
        slotgat_heads=None,
        slotgat_feat_drop=0.1,
        slotgat_attn_drop=0.1,
        slotgat_negative_slope=0.2,
        slotgat_residual=True,
        slotgat_alpha=0.1,
        slotgat_aggregator="average",
    ):
        super().__init__(config)

        self.num_labels = num_labels
        self.dataset = dataset
        self.use_structured_evidence = use_structured_evidence
        self.use_consistency_loss = use_consistency_loss
        self.use_relation_prior = use_relation_prior
        self.use_slotgat = use_slotgat
        self.consistency_loss_weight = consistency_loss_weight
        self.relation_prior_weight = relation_prior_weight
        self.node_dim = gtn_out * gtn_num_channels
        self.base_dim = self.node_dim * 3

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.etypes = ["dialog", "sequence", "entity", "speaker"]

        if slotgat_heads is None:
            slotgat_heads = [4, 4]

        if self.use_slotgat:
            self.node_dim = gtn_out * 3
            self.base_dim = self.node_dim * 3
            self.slotgat = SlotGATModule(
                edge_dim=slotgat_edge_dim,
                num_etypes=len(self.etypes),
                in_dims=config.hidden_size,
                num_hidden=slotgat_hidden_dim,
                num_classes=gtn_out * 3,
                num_layers=slotgat_num_layers,
                heads=slotgat_heads,
                feat_drop=slotgat_feat_drop,
                attn_drop=slotgat_attn_drop,
                negative_slope=slotgat_negative_slope,
                residual=slotgat_residual,
                alpha=slotgat_alpha,
                num_ntype=1,
                aggregator=slotgat_aggregator,
            )
        else:
            self.gtn = GTN(self.etypes, gtn_num_channels, config.hidden_size, gtn_out, gtn_num_layers)

        self.evidence_adapter = StructuredEvidenceAdapter(
            node_dim=self.node_dim,
            base_dim=self.base_dim,
            dropout=config.hidden_dropout_prob,
            evidence_bias_scale=evidence_bias_scale,
            use_confidence_gate=use_confidence_gate,
        )

        self.classifier = nn.Linear(self.base_dim, self.num_labels)

        self.relation_prior_gate = nn.Linear(1, 1)

        self.init_weights()

    def _build_consistency_loss(self, attention_probs, evidence_turn_mask, evidence_turn_scores, global_confidence):
        if evidence_turn_mask.sum() <= 0 or evidence_turn_scores.sum() <= 0:
            return None

        target = evidence_turn_scores * evidence_turn_mask
        target = target / target.sum().clamp(min=1e-8)

        pred = attention_probs.clamp(min=1e-8)
        kl = F.kl_div(pred.log(), target, reduction="batchmean")

        return kl * global_confidence.clamp(min=0.0, max=1.0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        speaker_ids=None,
        graphs=None,
        hidialog_mask=None,
        cls_indices=None,
        tran_ids=None,
        turn_ids=None,
        evidence_turn_mask=None,
        evidence_turn_scores=None,
        relation_prior=None,
        evidence_global_confidence=None,
        evidence_valid=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=hidialog_mask,
            token_type_ids=None,
            position_ids=position_ids,
            speaker_ids=speaker_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_outputs = outputs[0]
        graph_output = []
        attention_maps = []
        residual_gates = []
        consistency_losses = []

        for batch_idx, graph in enumerate(graphs):
            sequence_output = sequence_outputs[batch_idx]
            mask = cls_indices[batch_idx].unsqueeze(-1).expand(-1, sequence_output.shape[-1])
            features_graph = torch.masked_select(sequence_output, mask).view(-1, sequence_output.shape[-1])

            num_graph_nodes = graph.number_of_nodes('node') if hasattr(graph, 'number_of_nodes') else 0
            if num_graph_nodes > 0 and features_graph.shape[0] != num_graph_nodes:
                if features_graph.shape[0] < num_graph_nodes:
                    pad = torch.zeros(num_graph_nodes - features_graph.shape[0], features_graph.shape[-1], device=features_graph.device)
                    features_graph = torch.cat([features_graph, pad], dim=0)
                else:
                    features_graph = features_graph[:num_graph_nodes]

            if self.use_slotgat:
                slotgat_output, _ = self.slotgat(graph, features_graph, output_attention=output_attentions)
                if len(slotgat_output.shape) > 2:
                    slotgat_output = slotgat_output.flatten(1)
                if slotgat_output.shape[-1] == self.base_dim:
                    node_features = slotgat_output
                elif slotgat_output.shape[-1] == self.node_dim:
                    node_features = slotgat_output.repeat(1, 3)
                else:
                    if not hasattr(self, "slotgat_proj") or self.slotgat_proj.in_features != slotgat_output.shape[-1]:
                        self.slotgat_proj = nn.Linear(slotgat_output.shape[-1], self.base_dim).to(slotgat_output.device)
                    node_features = self.slotgat_proj(slotgat_output)

                dialog_node = node_features[0][: self.node_dim]
                obj_node = node_features[-2][: self.node_dim]
                subj_node = node_features[-1][: self.node_dim]
                base_feature = torch.cat((dialog_node, obj_node, subj_node), dim=-1)
                turn_node_count = max(int(graph.number_of_nodes("node")) - 2, 1)
                turn_features = node_features[:turn_node_count, : self.node_dim]
            else:
                graph_features, _, _ = self.gtn(graph, features_graph, output_gtn_adjacency=output_attentions)
                dialog_node = graph_features[0]
                obj_node = graph_features[-2]
                subj_node = graph_features[-1]
                base_feature = torch.cat((dialog_node, obj_node, subj_node), dim=-1)
                turn_node_count = max(int(graph.number_of_nodes("node")) - 2, 1)
                turn_features = graph_features[:turn_node_count]

            evidence_mask_i = evidence_turn_mask[batch_idx, :turn_node_count]
            evidence_scores_i = evidence_turn_scores[batch_idx, :turn_node_count]
            global_confidence_i = evidence_global_confidence[batch_idx].view(1)

            enhanced_feature, attention_probs, _, residual_gate = self.evidence_adapter(
                turn_features=turn_features,
                base_feature=base_feature,
                evidence_turn_mask=evidence_mask_i,
                evidence_turn_scores=evidence_scores_i,
                global_confidence=global_confidence_i,
                use_structured_evidence=self.use_structured_evidence,
            )

            if self.use_consistency_loss and evidence_valid is not None and evidence_valid[batch_idx] > 0:
                cons_loss = self._build_consistency_loss(
                    attention_probs,
                    evidence_mask_i,
                    evidence_scores_i,
                    evidence_global_confidence[batch_idx],
                )
                if cons_loss is not None:
                    consistency_losses.append(cons_loss)

            graph_output.append(enhanced_feature)
            attention_maps.append(attention_probs)
            residual_gates.append(residual_gate)

        pooled_output = self.dropout(torch.stack(graph_output))
        logits = self.classifier(pooled_output).view(-1, self.num_labels)

        if self.use_relation_prior and relation_prior is not None and evidence_global_confidence is not None:
            eta = torch.sigmoid(self.relation_prior_gate(evidence_global_confidence.unsqueeze(-1)))
            logits = logits + eta * self.relation_prior_weight * relation_prior

        loss = None
        if labels is not None and self.dataset in ["MRDA", "MELD"]:
            loss_fct = CrossEntropyLoss()
            labels = labels.argmax(dim=-1)
            loss = loss_fct(logits, labels)
        elif labels is not None and self.dataset not in ["MRDA", "MuTual", "MELD"]:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, self.num_labels)
            loss = loss_fct(logits, labels)
        elif labels is not None and self.dataset in ["MuTual"]:
            logits = logits.view(-1, 4)
            loss_fct = CrossEntropyLoss()
            labels = torch.argmax(labels.view(-1, 4), dim=1)
            loss = loss_fct(logits, labels)
            logits = logits.view(-1, self.num_labels)

        if consistency_losses:
            consistency_loss = torch.stack(consistency_losses).mean()
        else:
            consistency_loss = torch.tensor(0.0, device=logits.device)

        if loss is not None and self.use_consistency_loss:
            loss = loss + self.consistency_loss_weight * consistency_loss

        extras = {
            "consistency_loss": consistency_loss,
            "attention_maps": attention_maps,
            "residual_gates": residual_gates,
        }

        if labels is not None:
            return loss, logits, extras
        return logits, None, extras
