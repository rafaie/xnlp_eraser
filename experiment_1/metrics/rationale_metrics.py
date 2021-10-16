from typing import Union, Tuple, Dict, List, Optional

from allennlp.training.metrics import Metric
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import FBetaMultiLabelMeasure
from overrides import overrides
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score


class RationalMetricSingleSent(FBetaMultiLabelMeasure):
    def __init__(
        self,
        beta: float = 1.0,
        average: str = None,
        labels: List[int] = None,
        threshold: float = 0.5,
        num_classes: int = 1000
    ) -> None:
        super().__init__(beta, average, labels)
        self._threshold = threshold
        self.num_classes = num_classes
        self.max_token_size = 0

    @overrides
    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        token_size: int,
        mask: Optional[torch.BoolTensor] = None,
    ):
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(
                self.num_classes, device=predictions.device)
            self._true_sum = torch.zeros(
                self.num_classes, device=predictions.device)
            self._pred_sum = torch.zeros(
                self.num_classes, device=predictions.device)
            self._total_sum = torch.zeros(
                self.num_classes, device=predictions.device)

        self.max_token_size = max(self.max_token_size, token_size)
        super().__call__(predictions, gold_labels, mask)

    @overrides
    def get_metric(self, reset: bool = False):
        r = super().get_metric(reset)
        m = {
            'precision': np.mean(r['precision'][:self.max_token_size]),
            'recall': np.mean(r['recall'][:self.max_token_size]),
            'fscore': np.mean(r['fscore'][:self.max_token_size]),
        }

        if reset is True:
            self.max_token_size = 0

        return m


class RationalCntMetricSingleSent(FBetaMultiLabelMeasure):
    def __init__(
        self,
        beta: float = 1.0,
        average: str = None,
        labels: List[int] = None,
    ) -> None:
        super().__init__(beta, average, labels)
        self.mse_list = []

    @overrides
    def __call__(
        self,
        logits_rational_cnt: torch.Tensor,
        evidence_cnt_target: torch.Tensor
    ):
        self.mse_list += ((logits_rational_cnt -
                          evidence_cnt_target) ** 2).detach().tolist()

    @overrides
    def get_metric(self, reset: bool = False):
        mse = {
            'mse': np.mean(self.mse_list)}

        if reset:
            self.reset()

        return mse

    @overrides
    def reset(self) -> None:
        self.mse_list = []
