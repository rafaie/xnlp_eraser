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


@Model.register("simple_model_2docs2")
class SimpleModel2Docs2(BaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        premise_field_embedder: TextFieldEmbedder,
        query_field_embedder: TextFieldEmbedder,
        premise_seq2seq_encoder: Seq2SeqEncoder,
        query_seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: FeedForward,
        attention:Attention,
        need_2docs_attention:bool=True,
        premise_attention: Attention=None,
        query_attention: Attention=None,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):
        super(SimpleModel2Docs2, self).__init__(
            vocab, initializer, regularizer)
        self._vocabulary = vocab

        self._num_labels = self._vocabulary.get_vocab_size("labels")
        if self._num_labels == 0:
            self._num_labels = 5  # for COSE

        self._premise_field_embedder = premise_field_embedder
        self._query_field_embedder = query_field_embedder
        self._premise_seq2seq_encoder = premise_seq2seq_encoder
        self._query_seq2seq_encoder = query_seq2seq_encoder
        self._premise_dropout = torch.nn.Dropout(p=dropout)
        self._query_dropout = torch.nn.Dropout(p=dropout)

        if premise_attention is not None:
            self._premise_attention = premise_attention
            self._premise_vector = torch.nn.Parameter(torch.randn(
                (1, self._premise_seq2seq_encoder.get_output_dim())))

        if query_attention is not None:
            self._query_attention = query_attention
            self._query_vector = torch.nn.Parameter(torch.randn(
                (1, self._query_seq2seq_encoder.get_output_dim())))

        self._need_2docs_attention = need_2docs_attention
        self._attention = attention

        self._feedforward_encoder = feedforward_encoder
        self._classification_layer = torch.nn.Linear(
            self._feedforward_encoder.get_output_dim(),
            1)

        self._loss_tracks = {
            k: Average() for k in ["base_loss"]}

        initializer(self)

    def forward(self, document, premise, query,
                kept_tokens, premise_kept_tokens, query_kept_tokens,
                rationale=None, label=None, metadata=None) -> Dict[str, Any]:
        premise, query = self._regenerate_tokens_with_labels(
            metadata=metadata, labels=label)

        # process for premise
        premise_embedded_text = self._premise_field_embedder(premise)
        premise_mask = util.get_text_field_mask(premise).float()

        premise_embedded_text = self._premise_dropout(self._premise_seq2seq_encoder(premise_embedded_text, mask=premise_mask))

        if self._need_2docs_attention is True:
            premise_attentions = self._premise_attention(vector=self._premise_vector, matrix=premise_embedded_text, matrix_mask=premise_mask)
            premise_embedded_text = premise_embedded_text * premise_attentions.unsqueeze(-1) * premise_mask.unsqueeze(-1)

        # process for Query
        query_embedded_text = self._query_field_embedder(query)
        query_mask = util.get_text_field_mask(query).float()

        query_embedded_text = self._query_dropout(self._query_seq2seq_encoder(query_embedded_text, mask=query_mask))

        if self._need_2docs_attention is True:
            query_attentions = self._query_attention(vector=self._query_vector, matrix=query_embedded_text, matrix_mask=query_mask)
            query_embedded_text = query_embedded_text * query_attentions.unsqueeze(-1) * query_mask.unsqueeze(-1)

        # aggregate and generate the final layers

        # embedded_text = torch.cat(
        # (premise_embedded_text, query_embedded_text), 1)
        # embedded_vec = self._feedforward_encoder(embedded_text.sum(1))

        embedded_text = self._attention(premise_embedded_text, query_embedded_text)
        embedded_vec = self._feedforward_encoder(embedded_text.sum(2))

        logits = self._classification_layer(embedded_vec)
        b = logits.shape[0] // 5
        logits = logits.view(b, 5, 1)
        logits = logits.squeeze(-1)
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


    def _regenerate_tokens_with_labels(self, metadata, labels):
        tokens = [m["tokens"] for m in metadata]

        premise_instances = []
        query_instances = []
        new_tokens = []

        for words, meta, instance_labels in zip(tokens, metadata, labels):
            try :
                premises, queries = metadata[0]["convert_tokens_to_instance_2docs"](
                    words, [instance_labels[k] for k in ["A", "B", "C", "D", "E"]]
                )

                premise_instances += premises
                query_instances += queries
            except :
                breakpoint()

        premise_batch = Batch(premise_instances)
        premise_batch.index_instances(self._vocabulary)
        premise_padding_lengths = premise_batch.get_padding_lengths()
        premise_batch = premise_batch.as_tensor_dict(premise_padding_lengths)
        premises = {"bert": {k: v.to(next(self.parameters()).device) for k, v in premise_batch["document"]["bert"].items()}}

        query_batch = Batch(query_instances)
        query_batch.index_instances(self._vocabulary)
        query_padding_lengths = query_batch.get_padding_lengths()
        query_batch = query_batch.as_tensor_dict(query_padding_lengths)
        queries= {"bert": {k: v.to(next(self.parameters()).device) for k, v in query_batch["document"]["bert"].items()}}

        return premises, queries

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        base_metrics = super(SimpleModel2Docs2, self).get_metrics(reset)

        loss_metrics = {"_total" + k: v._total_value for k,
                        v in self._loss_tracks.items()}
        loss_metrics.update({k: v.get_metric(reset)
                            for k, v in self._loss_tracks.items()})
        loss_metrics.update(base_metrics)

        return loss_metrics