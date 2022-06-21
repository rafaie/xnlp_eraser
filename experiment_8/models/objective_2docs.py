from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from .base_model import BaseModel
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.attention import Attention

from allennlp.data.batch import Batch
from allennlp.training.metrics import Average
from .attentions import Attentions_dict


@Model.register("objective_2docs")
class Objective2Docs(BaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        premise_seq2seq_encoder: Seq2SeqEncoder,
        query_seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: FeedForward,
        att_class_name: str,
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
            self._classifier_input_dim, 1)

        self._attention = Attentions_dict[att_class_name](vocab=vocab)

        initializer(self)

    def forward(self, document, premise_text, premise_mask,
                query_text, query_mask, label=None, metadata=None, output_dict=None) -> Dict[str, Any]:

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
        b = logits.shape[0] // 5
        logits = logits.view(b, 5, 1)
        logits = logits.squeeze(-1)
        probs = torch.nn.Softmax(dim=-1)(logits)

        if output_dict is None:
            output_dict = {}

        if label is not None:
            if isinstance(label, list):
                output_labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                label = torch.LongTensor(
                    [output_labels[l["Label"]] for l in label]).to(logits.device)
            # label = F.one_hot(label, num_classes=self._num_labels)
            loss = F.cross_entropy(logits, label, reduction="none")
            output_dict["obj_loss"] = loss.mean()

        output_dict["logits"] = logits
        output_dict["probs"] = probs
        output_dict["class_probs"] = probs.max(-1)[0]
        output_dict["predicted_labels"] = probs.argmax(-1)
        output_dict["gold_labels"] = label
        output_dict["metadata"] = metadata

        self._call_metrics(output_dict)

        return output_dict

    # def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # base_metrics = super(Objective2Docs, self).get_metrics(reset)

        # loss_metrics = {"_total" + k: v._total_value for k,
        #                 v in self._loss_tracks.items()}
        # loss_metrics.update({k: v.get_metric(reset)
        #                     for k, v in self._loss_tracks.items()})
        # loss_metrics.update(base_metrics)

        # return loss_metrics
