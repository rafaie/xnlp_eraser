from typing import Optional, Dict, Union

import torch
import numpy as np
from torch.functional import Tensor
from torch.nn import functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, F1Measure, FBetaMultiLabelMeasure
from torch.nn import functional as F
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.fields import Field, ListField, MultiLabelField, SequenceLabelField
from experiment_1.metrics.rationale_metrics import RationalMetricSingleSent, RationalCntMetricSingleSent


ML_BASELINE = 'baseline'
ML_RATIONAL_TO_PRED = 'rational_to_pred'
ML_RATIONAL_CNT_TO_PRED = 'rational_cnt_to_pred'

@Model.register('fine_tune_baseline')
class FineTuneBaseline(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 serialization_dir: Optional[str],
                 regularizer: Optional[RegularizerApplicator] = None,
                 target_class_num: int = 3,
                 classifier: Union[StackedBidirectionalLstm, FeedForward] = None,
                 rational_mclassifier: FeedForward = None,
                 dropout: float = 0.0) -> None:
        super().__init__(vocab, regularizer=regularizer, serialization_dir=serialization_dir)
        self.embedder = embedder
        self.classifier = classifier or torch.nn.Linear(
            embedder.get_output_dim(), target_class_num)
        self.rational_mclassifier = rational_mclassifier or torch.nn.Linear(
            embedder.get_output_dim(), embedder.get_output_dim())
        self.dropout = torch.nn.Dropout(dropout)
        self.loss_fn_target = torch.nn.CrossEntropyLoss()
        self.loss_fn_rational = torch.nn.BCEWithLogitsLoss()

        self.acc = CategoricalAccuracy()
        self.acc_rational = RationalMetricSingleSent(
            num_classes=embedder.get_output_dim())
        self.num_rational_tokens = embedder.get_output_dim()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        m = self.acc_rational.get_metric(reset)
        return {
            'accuracy': self.acc.get_metric(reset),
            'rational_precision': m['precision'],
            'rational_recall': m['recall'],
            'rational_fscore': m['fscore'],
        }

    def gen_mask(self, t: Tensor) -> Tensor:
        n = self.num_rational_tokens
        device = t.device
        o = torch.ones(t.shape, device=device)
        z = torch.ones((t.shape[0], n - t.shape[1]), device=device)
        m = torch.cat((o, z), axis=1)
        return m.type(torch.bool)

    def complete_tensor(self, t: Tensor) -> Tensor:
        n = self.num_rational_tokens
        device = t.device
        z = torch.ones((t.shape[0], n - t.shape[1]), device=device)
        return torch.cat((t, z), axis=1)

    def get_evidences_dict(self, evidences:Tensor, meta:MetadataField):
        l = []

        for i, m in enumerate(meta):
            st = 0
            l2 = []
            for ii, k in enumerate(m['doc_ids']):
                d = {'docid': k, "hard_rationale_predictions": []}
                h = d["hard_rationale_predictions"]
                rat = []
                offset = m['offsets'][ii]
                for of in offset:
                    t = torch.sum(evidences[i][of[0] + st: of[1] + st + 1]) / (of[1] - of[0] + 1)
                    rat.append(torch.round(t))

                ev_st = -1
                for i, r in enumerate(rat):
                    if r == 1:
                        if ev_st == -1:
                            ev_st = i
                    else:
                        if ev_st != -1:
                            h.append({"start_token": ev_st, "end_token": i})
                            ev_st = -1
                if ev_st != -1:
                    h.append({"start_token": ev_st, "end_token": i})

                st = i + 1
                l2.append(d)
            l.append(l2)
        return l

# {"docid": "1061467336.jpg#3r1e_hypothesis", "hard_rationale_predictions": [{"start_token": 0, "end_token": 1}]}, 

    def forward(self, sent_query: TextField, evidences: TextField = None,
                label_target: LabelField = None,
                meta: MetadataField = None) -> Dict[str, torch.Tensor]:
        evidences = evidences.double()

        embedded_sent = self.embedder(sent_query)
        batch_size, num_tokens_per_sent, num_dim = embedded_sent.size()

        embedded_sent = embedded_sent[:, 0, :]
        embedded_sent = self.dropout(embedded_sent)

        logits = self.classifier(embedded_sent).squeeze().view(batch_size, -1)
        probs = F.softmax(logits, dim=1)

        logits_rational = self.rational_mclassifier(
            embedded_sent).squeeze().view(batch_size, -1)
        logits_rational = logits_rational[:, :num_tokens_per_sent]

        labels = meta[0]['labels']
        output_dict = {
            "annotation_id":[m['annotation_id'] for m in meta] ,
            "classification": [labels[m] for m in torch.argmax(logits, axis=1)],
            "classification_scores":logits,
            "logits_rational": logits_rational,
            "evidences_val": evidences,
            "evidences":self.get_evidences_dict(evidences, meta),
            "meta": meta
        }

        if label_target is not None:
            loss1 = self.loss_fn_target(logits, label_target)
            loss2 = self.loss_fn_rational(logits_rational,
                                          evidences)

            output_dict['loss'] = loss1 + loss2
            self.acc(probs, label_target)
            self.acc_rational(self.complete_tensor(logits_rational),
                              self.complete_tensor(evidences),
                              num_tokens_per_sent,
                              self.gen_mask(evidences))

        return output_dict

