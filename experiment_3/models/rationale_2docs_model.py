from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward

from allennlp.training.metrics import F1Measure, Average


@Model.register("rationale_2docs")
class Rationale2DocsModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        premise_field_embedder: TextFieldEmbedder,
        query_field_embedder: TextFieldEmbedder,
        premise_seq2seq_encoder: Seq2SeqEncoder,
        query_seq2seq_encoder: Seq2SeqEncoder,
        premise_feedforward_encoder: Seq2SeqEncoder,
        query_feedforward_encoder: Seq2SeqEncoder,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(Rationale2DocsModel, self).__init__(vocab, regularizer)
        self._vocabulary = vocab
        self._premise_field_embedder = premise_field_embedder
        self._query_field_embedder = query_field_embedder
        self._premise_seq2seq_encoder = premise_seq2seq_encoder
        self._query_seq2seq_encoder = query_seq2seq_encoder
        self._premise_dropout = torch.nn.Dropout(p=dropout)
        self._query_dropout = torch.nn.Dropout(p=dropout)

        self._premise_feedforward_encoder = premise_feedforward_encoder
        self._query_feedforward_encoder = query_feedforward_encoder
        # + query_feedforward_encoder.get_output_dim()
        self._classifier_input_dim = premise_feedforward_encoder.get_output_dim()

        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, 1)

        self._rationale_f1_metric = F1Measure(positive_label=1)
        self._rationale_length = Average()
        self._rationale_supervision_loss = Average()

        initializer(self)

    def forward(self, document, premise, query, rationale=None) -> Dict[str, Any]:
        # Premise
        premise_embedded_text = self._premise_field_embedder(premise)
        premise_mask = util.get_text_field_mask(premise).float()

        premise_embedded_text = self._premise_dropout(
            self._premise_seq2seq_encoder(premise_embedded_text, mask=premise_mask))
        premise_embedded_text = self._premise_feedforward_encoder(
            premise_embedded_text)

        # query
        query_embedded_text = self._query_field_embedder(query)
        query_mask = util.get_text_field_mask(query).float()

        query_embedded_text = self._query_dropout(
            self._query_seq2seq_encoder(query_embedded_text, mask=query_mask))
        query_embedded_text = self._query_feedforward_encoder(
            query_embedded_text)

        embedded_text = torch.cat(
            (premise_embedded_text, query_embedded_text), 1)
        logits = self._classification_layer(embedded_text).squeeze(-1)
        probs = torch.sigmoid(logits)

        output_dict = {}

        device=premise_embedded_text.device
        mask = util.get_text_field_mask(document).float()
        mask = torch.cat(
            (mask, torch.zeros((mask.shape[0], probs.shape[1]-mask.shape[1]), device=device)), 1)
        output_dict['mask'] = mask

        predicted_rationale = (probs > 0.5).long()
        output_dict['predicted_rationale'] = predicted_rationale * mask
        output_dict["prob_z"] = probs * mask

        premise_mask2 = torch.cat((premise_mask, torch.zeros(
            (mask.shape[0], mask.shape[1] - premise_mask.shape[1]), device=device)), 1)
        output_dict["premise_prob_z"] = (
            probs * premise_mask2)[:, :premise_mask.shape[1]]

        query_mask2 = mask - premise_mask2
        premise_len = premise_mask2.sum(1).int().tolist()
        query_prob_z_tmp = probs * query_mask2
        for i, l in enumerate(premise_len):
            if i == 0:
                query_prob_z = query_prob_z_tmp[i, l:l+query_mask.shape[1]].unsqueeze(0)
            else:
                query_prob_z = torch.cat(
                    (query_prob_z, query_prob_z_tmp[i, l:l+query_mask.shape[1]].unsqueeze(0)), 0)
        output_dict["query_prob_z"] = query_prob_z

        class_probs = torch.cat(
            [1 - probs.unsqueeze(-1), probs.unsqueeze(-1)], dim=-1)

        average_rationale_length = util.masked_mean(
            output_dict['predicted_rationale'], mask.bool(), dim=-1).mean()
        self._rationale_length(average_rationale_length.item())

        if rationale is not None:
            rationale = torch.cat((rationale, torch.zeros(
                (rationale.shape[0], mask.shape[1] - rationale.shape[1]), device=device)), 1)
            rationale_loss = F.binary_cross_entropy_with_logits(
                logits, rationale.float(), weight=mask)
            output_dict['rationale_supervision_loss'] = rationale_loss
            output_dict['gold_rationale'] = rationale * mask
            self._rationale_f1_metric(
                predictions=class_probs, gold_labels=rationale, mask=mask.bool())
            self._rationale_supervision_loss(rationale_loss.item())

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, f1 = self._rationale_f1_metric.get_metric(reset)
        # metrics = {'_rationale_' + k: v for v,
        #            k in zip([p, r, f1], ['p', 'r', 'f1'])}
        metrics = {}
        metrics.update(
            {'_rationale_length': self._rationale_length.get_metric(reset)})
        metrics.update(
            {'rationale_loss': self._rationale_supervision_loss.get_metric(reset)})

        return metrics
