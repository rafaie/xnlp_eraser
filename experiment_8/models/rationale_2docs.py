from typing import Optional, Dict, Any
from regex import P

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from .base_model import BaseModel
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.attention import Attention

from allennlp.data.batch import Batch
from allennlp.training.metrics import Average, F1Measure
from .attentions import Attentions_dict


@Model.register("rationale_2docs")
class Objective2Docs(BaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        premise_seq2seq_encoder: Seq2SeqEncoder,
        query_seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: FeedForward,
        att_class_name: str,
        agg_func: str,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):
        super(Objective2Docs, self).__init__(
            vocab, initializer, regularizer)

        self._num_labels = self._vocabulary.get_vocab_size("labels")
        if self._num_labels == 0:
            self._num_labels = 5  # for COSE

        self._premise_seq2seq_encoder = premise_seq2seq_encoder
        self._query_seq2seq_encoder = query_seq2seq_encoder

        self._premise_dropout = torch.nn.Dropout(p=dropout)
        self._query_dropout = torch.nn.Dropout(p=dropout)

        self._feedforward_encoder = feedforward_encoder

        self._classifier_input_dim = self._feedforward_encoder.get_output_dim()
        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, self._premise_seq2seq_encoder.get_input_dim())

        self._attention = Attentions_dict[att_class_name](vocab=vocab)
        self._agg_func = agg_func

        self._rationale_f1_metric = F1Measure(positive_label=1)
        self._rationale_length = Average()
        self._rationale_supervision_loss = Average()

        initializer(self)

    def get_agg_func(self, agg_func: str):
        return {
            'min': torch.min,
            'max': torch.max,
            'mean': torch.mean
        }[agg_func]

    def forward(self, document, premise_text, premise_mask,
                query_text, query_mask, rationale=None, metadata=None) -> Dict[str, Any]:

        device = premise_text.device

        # Process premise
        premise_text = self._premise_dropout(
            self._premise_seq2seq_encoder(premise_text, mask=premise_mask))

        # Process query
        query_text = self._query_dropout(
            self._query_seq2seq_encoder(query_text, mask=query_mask))

        # Resize query
        # TODO: Update Query or premises based on tnspr size
        batch_size, premis_len, hidden_len = premise_text.shape
        query_len = query_text.shape[1]
        query_text = torch.cat((query_text,
                               torch.zeros((batch_size, premis_len - query_len, hidden_len),
                                           device=premise_text.device)),
                               axis=1)
        query_mask = torch.cat((query_mask,
                               torch.zeros((batch_size, premis_len - query_len),
                                           device=premise_text.device)),
                               axis=1)

        attentions = self._attention(query_states=premise_text,
                                     attention_mask=premise_mask,
                                     source_states=query_text,
                                     source_attention_mask=query_mask)

        embedded_text = premise_text * \
            attentions.hidden_states.sum(-1).unsqueeze(-1) * \
            premise_mask.unsqueeze(-1)
        embedded_vec = self._feedforward_encoder(embedded_text.sum(1))

        logits = self._classification_layer(embedded_vec)
        premise_len = premise_mask.sum(1).int().tolist()
        l_max = logits.shape[1]

        for i, l in enumerate(premise_len):
            p = torch.cat((logits[i, :l], torch.zeros(l_max-l, device=device)), 0).unsqueeze(0)
            q = torch.cat((logits[1, l:], torch.zeros(l, device=device)), 0).unsqueeze(0)
            if i == 0:
                premise_logit = p
                query_logit = q
            else:
                premise_logit = torch.cat((premise_logit, p), 0)
                query_logit = torch.cat((query_logit, q), 0)


        # rationale_len = rationale.shape[1]
        # logits = logits[:, :rationale_len]
        p_max = premise_logit.shape[1]
        b = premise_logit.shape[0] // 5
        logits = premise_logit.view(b, 5, p_max)
        logits = self.get_agg_func(self._agg_func)(logits, 1)

        p2_max = premise_mask.shape[1]
        b2 = premise_mask.shape[0] // 5
        premise_mask2 = premise_mask.view(b2, 5, p2_max)
        premise_mask2 = self.get_agg_func(self._agg_func)(premise_mask2, 1)

        probs = torch.sigmoid(logits)
        class_probs = torch.cat(
            [1 - probs.unsqueeze(-1), probs.unsqueeze(-1)], dim=-1)

        output_dict = {}
        if rationale is not None:
            max_r = rationale.shape[1]
            loss = torch.nn.BCEWithLogitsLoss()(logits[:,:max_r], rationale.float())
            output_dict["rat_loss"] = loss.mean()
            self._rationale_f1_metric(
                predictions=class_probs[:, :max_r], gold_labels=rationale, mask=premise_mask2[:, :max_r].bool())
            self._rationale_supervision_loss(loss.item())

        output_dict["rat_logits"] = logits
        output_dict["rat_probs"] = probs
        output_dict["rat_class_probs"] = probs.max(-1)[0]
        output_dict["rat_predicted_labels"] = probs.argmax(-1)
        output_dict["rat_gold_labels"] = rationale
        output_dict["rat_metadata"] = metadata
        output_dict["premise_logit"] = premise_logit
        output_dict["query_logit"] = query_logit

        # self._call_metrics(output_dict)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        m = self._rationale_f1_metric.get_metric(reset)
        metrics = {"_rationale_" + k: m[k] for k in m}
        metrics.update(
            {'_rationale_length': self._rationale_length.get_metric(reset)})
        metrics.update(
            {'rationale_loss': self._rationale_supervision_loss.get_metric(reset)})

        return metrics
