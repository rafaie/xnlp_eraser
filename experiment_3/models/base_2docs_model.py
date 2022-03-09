from typing import Optional, Dict, Any

import torch
import torch.distributions as D
import numpy as np

from allennlp.models.model import Model
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from experiment_3.models.base_model import BaseModel
from experiment_3.models.rationale_2docs_model import Rationale2DocsModel
from allennlp.training.metrics import Average


@Model.register("base_2docs")
class Base2DocsModel(BaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        rationale_model_params: Params,
        objective_model_params: Params,
        reg_loss_lambda: float,
        reg_loss_mu: float = 2,
        reinforce_loss_weight: float = 1.0,
        rationale_supervision_loss_weight: float = 1.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        loss_mode: str = BaseModel.LOSS_MODE_ALL
    ):

        super(Base2DocsModel, self).__init__(
            vocab, initializer, regularizer, loss_mode)
        self._vocabulary = vocab
        self._num_labels = self._vocabulary.get_vocab_size("labels")

        self._rationale_model = Model.from_params(
            vocab=vocab, regularizer=regularizer, initializer=initializer, params=Params(
                rationale_model_params)
        )
        self._objective_model = Model.from_params(
            vocab=vocab, regularizer=regularizer, initializer=initializer, params=Params(
                objective_model_params)
        )

        self._reg_loss_lambda = reg_loss_lambda
        self._reg_loss_mu = reg_loss_mu
        self._reinforce_loss_weight = reinforce_loss_weight
        self._rationale_supervision_loss_weight = rationale_supervision_loss_weight
        self._loss_tracks = {
            k: Average() for k in ["premise_lasso_loss", "premise_fused_lasso_loss",
                                   "query_lasso_loss", "query_fused_lasso_loss",
                                   "base_loss"]}

        initializer(self)

    def forward(self, document, premise, query,
                kept_tokens, premise_kept_tokens, query_kept_tokens,
                rationale=None, label=None, metadata=None) -> Dict[str, Any]:
        rationale_dict = self._rationale_model(
            document, premise, query, rationale)
        assert "premise_prob_z" in rationale_dict
        assert "query_prob_z" in rationale_dict

        mask = rationale_dict["mask"]

        premise_prob_z = rationale_dict["premise_prob_z"]
        premise_mask = util.get_text_field_mask(premise)
        assert len(premise_prob_z.shape) == 2
        premise_prob_z = premise_kept_tokens.float() + premise_prob_z * \
            (1 - premise_kept_tokens)
        premise_sampler = D.bernoulli.Bernoulli(probs=premise_prob_z)
        premise_sample_z = premise_sampler.sample() * premise_mask.float()

        query_prob_z = rationale_dict["query_prob_z"]
        query_mask = util.get_text_field_mask(query)
        assert len(query_prob_z.shape) == 2
        query_prob_z = query_kept_tokens.float() + query_prob_z * \
            (1 - query_kept_tokens)
        query_sampler = D.bernoulli.Bernoulli(probs=query_prob_z)
        query_sample_z = query_sampler.sample() * query_mask.float()

        objective_dict = self._objective_model(premise_sample_z=premise_sample_z,
                                               query_sample_z=query_sample_z,
                                               label=label, metadata=metadata)

        loss = 0.0

        if label is not None:
            assert "loss" in objective_dict

            loss_sample = objective_dict["loss"]  # (B,)
            if self.loss_mode != BaseModel.LOSS_MODE_RATIONALE_ONLY:
                loss += loss_sample.mean()

            # Permise
            premise_lasso_loss = util.masked_mean(
                premise_sample_z, premise_mask, dim=-1)  # (B,)
            premise_masked_sum = premise_mask[:, :-1].sum(-1).clamp(1e-5)
            premise_diff = (
                premise_sample_z[:, 1:] - premise_sample_z[:, :-1]).abs()
            premise_masked_diff = (premise_diff * premise_mask[:, :-1]).sum(-1)
            premise_fused_lasso_loss = premise_masked_diff / premise_masked_sum

            self._loss_tracks["premise_lasso_loss"](
                premise_lasso_loss.mean().item())
            self._loss_tracks["premise_fused_lasso_loss"](
                premise_fused_lasso_loss.mean().item())

            premise_log_prob_z = torch.log(
                1 + torch.exp(premise_sampler.log_prob(premise_sample_z)))  # (B, L)
            premise_log_prob_z_sum = (
                premise_mask * premise_log_prob_z).mean(-1)  # (B,)

            premise_generator_loss = (
                loss_sample.detach()
                + premise_lasso_loss * self._reg_loss_lambda
                + premise_fused_lasso_loss *
                (self._reg_loss_mu * self._reg_loss_lambda)
            ) * premise_log_prob_z_sum

            # Query
            query_lasso_loss = util.masked_mean(
                query_sample_z, query_mask, dim=-1)  # (B,)
            query_masked_sum = query_mask[:, :-1].sum(-1).clamp(1e-5)
            query_diff = (query_sample_z[:, 1:] - query_sample_z[:, :-1]).abs()
            query_masked_diff = (query_diff * query_mask[:, :-1]).sum(-1)
            query_fused_lasso_loss = query_masked_diff / query_masked_sum

            self._loss_tracks["query_lasso_loss"](
                query_lasso_loss.mean().item())
            self._loss_tracks["query_fused_lasso_loss"](
                query_fused_lasso_loss.mean().item())

            query_log_prob_z = torch.log(
                1 + torch.exp(query_sampler.log_prob(query_sample_z)))  # (B, L)
            query_log_prob_z_sum = (
                query_mask * query_log_prob_z).mean(-1)  # (B,)

            query_generator_loss = (
                loss_sample.detach()
                + query_lasso_loss * self._reg_loss_lambda
                + query_fused_lasso_loss *
                (self._reg_loss_mu * self._reg_loss_lambda)
            ) * query_log_prob_z_sum
            query_generator_loss_mean = query_generator_loss.mean() if np.isnan(query_generator_loss.mean(
            ).item()) is False else torch.as_tensor(0.0, device=premise_generator_loss.device)

            self._loss_tracks["base_loss"](loss_sample.mean().item())

            if self.loss_mode != BaseModel.LOSS_MODE_OBJECTIVE_ONLY:
                loss += self._reinforce_loss_weight * premise_generator_loss.mean() +  \
                    self._reinforce_loss_weight * query_generator_loss_mean

        output_dict = rationale_dict
        if self.loss_mode != BaseModel.LOSS_MODE_OBJECTIVE_ONLY:
            loss += self._rationale_supervision_loss_weight * \
            rationale_dict.get("rationale_supervision_loss", 0.0)

        output_dict["logits"] = objective_dict["logits"]
        output_dict['probs'] = objective_dict['probs']
        output_dict["class_probs"] = objective_dict["class_probs"]
        output_dict["predicted_labels"] = objective_dict["predicted_labels"]
        output_dict["gold_labels"] = objective_dict["gold_labels"]

        output_dict["loss"] = loss
        output_dict["metadata"] = metadata
        output_dict["mask"] = mask

        self._call_metrics(output_dict)

        return output_dict

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

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        base_metrics = super(Base2DocsModel, self).get_metrics(reset)

        loss_metrics = {"_total" + k: v._total_value for k,
                        v in self._loss_tracks.items()}
        loss_metrics.update({k: v.get_metric(reset)
                            for k, v in self._loss_tracks.items()})
        loss_metrics.update(base_metrics)
        loss_metrics.update(self._rationale_model.get_metrics(reset))

        return loss_metrics
