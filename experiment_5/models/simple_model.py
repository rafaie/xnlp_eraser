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



@Model.register("simple_model")
class SimpleModel(BaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        doc_field_embedder: TextFieldEmbedder,
        aggregation_layer:Seq2SeqEncoder,
        feedforward_encoder:FeedForward,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):
        super(SimpleModel, self).__init__(
            vocab, initializer, regularizer)
        self._vocabulary = vocab

        self._num_labels = self._vocabulary.get_vocab_size("labels")
        if self._num_labels == 0:
            self._num_labels = 5 # for COSE

        self._doc_field_embedder = doc_field_embedder
        self._aggregation_layer = aggregation_layer
        self._feedforward_encoder = feedforward_encoder
        self._dropout = torch.nn.Dropout(p=dropout)
        self._classification_layer = torch.nn.Linear(
            self._feedforward_encoder.get_output_dim(),
            self._num_labels)


        self._loss_tracks = {
            k: Average() for k in ["base_loss"]}

        initializer(self)

    def forward(self, document, premise, query,
                kept_tokens, premise_kept_tokens, query_kept_tokens,
                rationale=None, label=None, metadata=None) -> Dict[str, Any]:
        embedded_text = self._doc_field_embedder(document)
        mask = util.get_text_field_mask(document).float()

        embedded_text = self._aggregation_layer(embedded_text, mask=mask)
        embedded_text = self._feedforward_encoder(embedded_text.sum(1))
        embedded_text = self._dropout(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.Softmax(dim=-1)(logits)

        output_dict = {}

        if label is not None:
            if isinstance(label, list):
                output_labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                label = torch.LongTensor(
                    [output_labels[l["Label"]] for l in label]).to(logits.device)
            # label = F.one_hot(label, num_classes=self._num_labels)
            loss = F.cross_entropy(logits, label, reduction="none")
            output_dict["loss"] = loss.mean()

        output_dict["logits"] = logits
        output_dict["probs"] = probs
        output_dict["class_probs"] = probs.max(-1)[0]
        output_dict["predicted_labels"] = probs.argmax(-1)
        output_dict["gold_labels"] = label
        output_dict["metadata"] = metadata

        self._call_metrics(output_dict)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        base_metrics = super(SimpleModel, self).get_metrics(reset)

        loss_metrics = {"_total" + k: v._total_value for k,
                        v in self._loss_tracks.items()}
        loss_metrics.update({k: v.get_metric(reset)
                            for k, v in self._loss_tracks.items()})
        loss_metrics.update(base_metrics)

        return loss_metrics