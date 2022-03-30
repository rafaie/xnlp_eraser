from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from zmq import device
from .base_model import BaseModel
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.attention import Attention

from allennlp.data.batch import Batch
from allennlp.training.metrics import Average


@Model.register("model_2docs4")
class Model2Docs4(BaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        premise_field_embedder: TextFieldEmbedder,
        query_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: FeedForward,
        attention: Attention,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):
        super(Model2Docs4, self).__init__(
            vocab, initializer, regularizer)
        self._vocabulary = vocab

        self._num_labels = self._vocabulary.get_vocab_size("labels")
        if self._num_labels == 0:
            self._num_labels = 5  # for COSE

        self._premise_field_embedder = premise_field_embedder
        self._query_field_embedder = query_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._dropout = torch.nn.Dropout(p=dropout)

        self._attention = attention
        self._feedforward_encoder = feedforward_encoder

        self._classifier_input_dim = self._feedforward_encoder.get_output_dim()
        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, 1)

        self._vector = torch.nn.Parameter(torch.randn(
            (1, self._seq2seq_encoder.get_output_dim())))

        self._loss_tracks = {
            k: Average() for k in ["base_loss"]}

        initializer(self)

    # def forward(self, document, premise, query,
    #             kept_tokens, premise_kept_tokens, query_kept_tokens,
    #             rationale=None, label=None, metadata=None) -> Dict[str, Any]:
    def forward(self, document, kept_tokens, rationale=None, label=None, metadata=None) -> Dict[str, Any]:

        premise = self._regenerate_tokens_with_labels(
            metadata=metadata, labels=label)
        premise_text = self._premise_field_embedder(premise)
        premise_mask = util.get_text_field_mask(premise).float()

        query = self._regenerate_queries(metadata=metadata, labels=label)
        query_text = self._query_field_embedder(query)
        query_mask = util.get_text_field_mask(query).float()

        # Maerge them seperately
        batch_size = premise_mask.shape[0]
        str_len = premise_mask.shape[1] + query_mask.shape[1]
        embedded_text = premise_text
        mask = premise_mask
        # for i in range(batch_size):
        #     prem_len = int(premise_mask[i].sum().item())
        #     query_len = int(query_mask[i].sum().item())
        #     a = torch.cat((premise_text[i][:prem_len],
        #                    query_text[i][:query_len],
        #                    torch.zeros(str_len - prem_len - query_len,
        #                                premise_text.shape[-1], device=premise_text.device)
        #                    )).unsqueeze(0)
        #     m = torch.cat((
        #         torch.ones(prem_len + query_len, device=premise_text.device),
        #         torch.zeros(str_len - prem_len - query_len, device=premise_text.device)
        #     )).unsqueeze(0)
        #     if embedded_text == None:
        #         embedded_text = a
        #         mask = m
        #     else:
        #         embedded_text = torch.cat((embedded_text, a), 0)
        #         mask = torch.cat((mask, m), 0)

        # embedded_text = torch.cat((premise_text, query_text), 1)
        # mask = torch.cat((premise_mask, query_mask), 1)

        embedded_text = self._dropout(
            self._seq2seq_encoder(embedded_text, mask=mask))
        attentions = self._attention(
            vector=self._vector, matrix=embedded_text, matrix_mask=mask)

        embedded_text = embedded_text * \
            attentions.unsqueeze(-1) * mask.unsqueeze(-1)
        embedded_vec = self._feedforward_encoder(embedded_text.sum(1))

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
        # sample_z_cpu = sample_z.cpu().data.numpy()
        tokens = [m["tokens"] for m in metadata]

        # assert len(tokens) == len(sample_z_cpu)
        # assert max([len(x) for x in tokens]) == sample_z_cpu.shape[1]

        instances = []
        new_tokens = []
        for words, meta, instance_labels in zip(tokens, metadata, labels):
            # mask = mask[: len(words)]
            new_words = [w for i, w in enumerate(words)]

            new_tokens.append(new_words)
            meta["new_tokens"] = new_tokens
            try:
                instances += metadata[0]["convert_tokens_to_instance"](
                    new_words, [instance_labels[k]
                                for k in ["A", "B", "C", "D", "E"]]
                )
            except:
                breakpoint()

        batch = Batch(instances)
        batch.index_instances(self._vocabulary)
        padding_lengths = batch.get_padding_lengths()

        batch = batch.as_tensor_dict(padding_lengths)
        return {"bert": {k: v.to(next(self.parameters()).device) for k, v in batch["document"]["bert"].items()}}

    def _regenerate_queries(self, metadata, labels):
        instances = []
        new_tokens = []

        tokens = [m["tokens"] for m in metadata]

        instances = []
        new_tokens = []
        for words, meta, instance_labels in zip(tokens, metadata, labels):
            new_words = [w for i, w in enumerate(words)]

            new_tokens.append(new_words)
            meta["new_tokens"] = new_tokens
            try:
                instances += metadata[0]["convert_tokens_to_instance"](
                    [], [instance_labels[k] for k in ["A", "B", "C", "D", "E"]]
                )
            except:
                breakpoint()

        batch = Batch(instances)
        batch.index_instances(self._vocabulary)
        padding_lengths = batch.get_padding_lengths()

        batch = batch.as_tensor_dict(padding_lengths)
        return {"bert": {k: v.to(next(self.parameters()).device) for k, v in batch["document"]["bert"].items()}}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        base_metrics = super(Model2Docs4, self).get_metrics(reset)

        loss_metrics = {"_total" + k: v._total_value for k,
                        v in self._loss_tracks.items()}
        loss_metrics.update({k: v.get_metric(reset)
                            for k, v in self._loss_tracks.items()})
        loss_metrics.update(base_metrics)

        return loss_metrics
