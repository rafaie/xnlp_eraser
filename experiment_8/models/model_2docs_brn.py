from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import torch.distributions as D

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

TRANSFER_DIRECT = 0
TRANSFER_VIA_MUL = 1
TRANSFER_VIA_RESEDUAL = 2
TRANSFER_VIA_RESEDUAL_MUL = 3


@Model.register("model_2docs_brn")
class Model2DocsBrn(BaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        premise_field_embedder: TextFieldEmbedder,
        query_field_embedder: TextFieldEmbedder,
        rationale_model_params: Params,
        objective_model_params: Params,
        initializer: InitializerApplicator = InitializerApplicator(),
        transfer_method: int = TRANSFER_VIA_MUL,
        regularizer: Optional[RegularizerApplicator] = None,
    ):
        super(Model2DocsBrn, self).__init__(
            vocab, initializer, regularizer)
        self._vocabulary = vocab

        self._num_labels = self._vocabulary.get_vocab_size("labels")
        if self._num_labels == 0:
            self._num_labels = 5  # for COSE

        self._premise_field_embedder = premise_field_embedder
        self._query_field_embedder = query_field_embedder

        self._rationale_model = Model.from_params(
            vocab=vocab, regularizer=regularizer, initializer=initializer, params=Params(
                rationale_model_params)
        )

        self._objective_model = Model.from_params(
            vocab=vocab, regularizer=regularizer, initializer=initializer, params=Params(
                objective_model_params)
        )

        self.transfer_method = transfer_method

        self._loss_tracks = {
            k: Average() for k in ["base_loss"]}

        initializer(self)

    def forward(self, document, kept_tokens, rationale=None, label=None, metadata=None) -> Dict[str, Any]:

        device = kept_tokens.device

        # Process premise
        premise = self._regenerate_tokens_with_labels(
            metadata=metadata, labels=label)
        premise_text = self._premise_field_embedder(premise)
        premise_mask = util.get_text_field_mask(premise).float()

        # Process query
        query = self._regenerate_queries(metadata=metadata, labels=label)
        query_text = self._query_field_embedder(query)
        query_mask = util.get_text_field_mask(query).float()

        # Call rationale model
        rationaledict = self._rationale_model(
            document, premise_text, premise_mask, query_text, query_mask, rationale, metadata)

        # calculate new tesnor with rationale output
        # premise_logit = rationaledict['premise_logit']
        premise_prob_z = rationaledict["premise_prob_z"]
        kept_tokens2 = torch.cat((kept_tokens, torch.zeros(
            (kept_tokens.shape[0], premise_text.shape[1] - kept_tokens.shape[1]), device=device)), 1)
        kept_tokens2 = kept_tokens2.repeat(1, 5).view(premise_prob_z.shape)
        premise_prob_z = kept_tokens2.float() + premise_prob_z * (1 - kept_tokens2)
        premise_sampler = D.bernoulli.Bernoulli(probs=premise_prob_z)
        premise_sample_z = premise_sampler.sample() * premise_mask.float()
        # premise_text = premise_sample_z.unsqueeze(2) * premise_text
        if self.transfer_method == TRANSFER_DIRECT:
            premise_text = premise_sample_z.unsqueeze(2).repeat(1, 1, premise_text.shape[-1])
        elif self.transfer_method == TRANSFER_VIA_MUL:
            premise_text = premise_sample_z.unsqueeze(2) * premise_text
        elif self.transfer_method == TRANSFER_VIA_RESEDUAL:
            premise_sample_z = premise_sample_z.unsqueeze(2).repeat(1, 1, premise_text.shape[-1])
            premise_text = premise_sample_z + premise_text
        elif self.transfer_method == TRANSFER_VIA_RESEDUAL_MUL:
            premise_text2 = premise_sample_z.unsqueeze(2) * premise_text
            premise_text = premise_text2 + premise_text
        else:
            raise "not implemented"

        # premise_logit = torch.mul(
        #     premise_logit[:, :premise_mask.shape[1]], premise_mask.bool())
        # premise_text = premise_logit.unsqueeze(2) * premise_text

        # query_logit = rationaledict['query_logit']
        # query_logit = torch.mul(
        #     query_logit[:, :query_mask.shape[1]], query_mask.bool())
        # # query_text = query_logit.unsqueeze(2) * query_text # We can't apply it for COSE dataset!

        # Call objective model
        objective_dict = self._objective_model(
            document, premise_text, premise_mask, query_text,
            query_mask, label, metadata,
            output_dict=rationaledict)

        # calculate loss
        objective_dict['loss'] = objective_dict['rat_loss'] + \
            objective_dict['obj_loss']

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
        base_metrics = super(Model2DocsBrn, self).get_metrics(reset)

        loss_metrics = {"_total" + k: v._total_value for k,
                        v in self._loss_tracks.items()}
        loss_metrics.update({k: v.get_metric(reset)
                            for k, v in self._loss_tracks.items()})
        loss_metrics.update(base_metrics)
        loss_metrics.update(self._rationale_model.get_metrics(reset))

        return loss_metrics

    def _decode(self, output_dict) -> Dict[str, Any]:
        new_output_dict = {}

        output_dict["predicted_labels"] = output_dict["predicted_labels"].cpu(
        ).data.numpy()

        masks = output_dict["mask"].float().cpu().data.numpy()
        predicted_rationales = output_dict["predicted_rationale"].cpu(
        ).data.numpy()
        metadata = output_dict["metadata"]
        soft_scores = output_dict["prob_z"].cpu().data.numpy()

        new_output_dict["rationales"] = []

        for rationale, ss, mask, m in zip(predicted_rationales, soft_scores, masks, metadata):
            rationale = rationale[mask == 1]
            ss = ss[mask == 1]

            document_to_span_map = m["document_to_span_map"]
            document_rationale = []
            for docid, (s, e) in document_to_span_map.items():
                doc_rationale = list(rationale[s:e]) + [0]
                starts = []
                ends = []
                for i in range(len(doc_rationale) - 1):
                    if (doc_rationale[i - 1], doc_rationale[i]) == (0, 1):
                        starts.append(i)
                    if (doc_rationale[i], doc_rationale[i + 1]) == (1, 0):
                        ends.append(i + 1)

                spans = zip(starts, ends)
                document_rationale.append(
                    {
                        "docid": docid,
                        "hard_rationale_predictions": [{"start_token": s, "end_token": e} for s, e in list(spans)],
                    }
                )

            new_output_dict["rationales"].append(document_rationale)

        output_labels = self._vocabulary.get_index_to_token_vocabulary(
            "labels")
        if ('A' in list(output_labels.values()) and len(list(output_labels.values())) == 5) or \
                len(list(output_labels.values())) == 0:
            output_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

        new_output_dict["annotation_id"] = [m["annotation_id"]
                                            for m in metadata]
        new_output_dict["classification"] = [
            output_labels[int(p)] for p in output_dict["predicted_labels"]]

        _output_labels = [output_labels[i] for i in range(self._num_labels)]
        new_output_dict["classification_scores"] = [
            dict(zip(_output_labels, list(x))) for x in output_dict["probs"].cpu().data.numpy()
        ]

        return new_output_dict
