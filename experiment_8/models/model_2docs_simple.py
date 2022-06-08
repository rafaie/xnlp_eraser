from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.common.params import Params
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from zmq import device
from .base_model import BaseModel
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.attention import Attention
from allennlp.modules.transformer.attention_module import AttentionModule

from allennlp.data.batch import Batch
from allennlp.training.metrics import Average
from .attentions import Attentions_dict

import importlib
import os


@Model.register("model_2docs_simple")
class Model2DocsSimple(BaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        premise_field_embedder: TextFieldEmbedder,
        query_field_embedder: TextFieldEmbedder,
        objective_model_params: Params,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):
        super(Model2DocsSimple, self).__init__(
            vocab, initializer, regularizer)
        self._vocabulary = vocab

        self._num_labels = self._vocabulary.get_vocab_size("labels")
        if self._num_labels == 0:
            self._num_labels = 5  # for COSE

        self._premise_field_embedder = premise_field_embedder
        self._query_field_embedder = query_field_embedder

        self._objective_model = Model.from_params(
            vocab=vocab, regularizer=regularizer, initializer=initializer, params=Params(
                objective_model_params)
        )

        self._loss_tracks = {
            k: Average() for k in ["base_loss"]}

        initializer(self)

    def forward(self, document, kept_tokens, rationale=None, label=None, metadata=None) -> Dict[str, Any]:

        # Process premise
        premise = self._regenerate_tokens_with_labels(
            metadata=metadata, labels=label)
        premise_text = self._premise_field_embedder(premise)
        premise_mask = util.get_text_field_mask(premise).float()

        # Process query
        query = self._regenerate_queries(metadata=metadata, labels=label)
        query_text = self._query_field_embedder(query)
        query_mask = util.get_text_field_mask(query).float()

        # Call objective class
        objective_dict = self._objective_model(
            document, premise_text, premise_mask, query_text, query_mask, label, metadata)

        self._call_metrics(objective_dict)

        return objective_dict

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
        base_metrics = super(Model2DocsSimple, self).get_metrics(reset)

        loss_metrics = {"_total" + k: v._total_value for k,
                        v in self._loss_tracks.items()}
        loss_metrics.update({k: v.get_metric(reset)
                            for k, v in self._loss_tracks.items()})
        loss_metrics.update(base_metrics)

        return loss_metrics
