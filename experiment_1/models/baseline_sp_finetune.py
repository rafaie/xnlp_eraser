from typing import Optional, Dict, Union

import torch
import numpy as np
from torch import tensor
from torch.functional import Tensor
from torch.nn import functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, feedforward
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
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

RATIONAL_CAST_CUT = 'cut'

ACTIVATION = {
    'relu': torch.nn.ReLU(),
    'sigmoid': torch.nn.Sigmoid(),
    'tanh': torch.nn.Tanh()
}

@Model.register('fine_tune_baseline_sp')
class FineTuneBaseline(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 serialization_dir: Optional[str],
                 regularizer: Optional[RegularizerApplicator] = None,
                 target_class_num: int = 3,
                 encoder:Union[StackedBidirectionalLstm,
                                   FeedForward] = None,
                 classifier: FeedForward = None,
                 rational_mclassifier: FeedForward = None,
                 dropout: float = 0.0,
                 rational_cast_method: str = RATIONAL_CAST_CUT,
                 loss_co1: int = 1,
                 loss_co2: int = 1,
                 loss_co3: int = 1,
                 loss_b: int = 0) -> None:
        super().__init__(vocab, regularizer=regularizer, serialization_dir=serialization_dir)
        self.embedder = embedder
        self.encoder = encoder or torch.nn.Linear(
            embedder.get_output_dim(), embedder.get_output_dim())
        self.encoder_dict={}
        if isinstance(self.encoder, Dict):
            self.encoder_dict = self.encoder
            if self.encoder['type'] == 'feedforward':
                self.encoder=  FeedForward(
                        input_dim=self.encoder['input_dim'],
                        activations=ACTIVATION[self.encoder['activations']],
                        hidden_dims=self.encoder['hidden_dims'],
                        num_layers=self.encoder['num_layers'],
                        dropout=self.encoder['dropout'] if 'dropout' in list(self.encoder.keys()) else 0
                        )
            elif self.encoder['type'] == 'LSTM':
                self.encoder=  torch.nn.LSTM(
                        input_size=self.encoder['input_size'],
                        hidden_size=self.encoder['hidden_size'],
                        num_layers=self.encoder['num_layers'],
                        bias=self.encoder['bias'] if 'bias' in list(self.encoder.keys()) else True,
                        batch_first=self.encoder['batch_first'] if 'batch_first' in list(self.encoder.keys()) else False,
                        dropout=self.encoder['dropout'] if 'dropout' in list(self.encoder.keys()) else 0,
                        bidirectional=self.encoder['bidirectional'] if 'bidirectional' in list(self.encoder.keys()) else False,
                        proj_size=self.encoder['proj_size'] if 'proj_size' in list(self.encoder.keys()) else 0,
                        )


        self.classifier = classifier or torch.nn.Linear(
            embedder.get_output_dim(), target_class_num)
        self.rational_mclassifier = rational_mclassifier or torch.nn.Linear(
            embedder.get_output_dim(), embedder.get_output_dim())
        self.dropout = torch.nn.Dropout(dropout)
        self.loss_fn_target = torch.nn.CrossEntropyLoss()
        self.loss_fn_rational = torch.nn.BCEWithLogitsLoss()
        self.rational_cast_method = rational_cast_method

        self.acc = CategoricalAccuracy()
        self.acc_rational = RationalMetricSingleSent(
            num_classes=embedder.get_output_dim())
        self.num_rational_tokens = embedder.get_output_dim()
        self.loss_co1 = loss_co1
        self.loss_co2 = loss_co2
        self.loss_co3 = loss_co3
        self.loss_b = loss_b

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

    def get_evidences_dict(self, evidences: Tensor, meta: MetadataField):
        l = []

        for i, m in enumerate(meta):
            l2 = []
            for ii, k in enumerate(m['doc_ids']):
                d = {'docid': k, "hard_rationale_predictions": []}
                h = d["hard_rationale_predictions"]
                rat = []
                offset = m['offsets'][ii]
                mid_token_pos = 0 if ii == 0 else m['mid_token_pos']
                for of in offset:
                    t = torch.max(
                        evidences[i][of[0] + mid_token_pos: of[1] + mid_token_pos + 1])
                    rat.append(torch.round(t))

                ev_st = -1
                for iii, r in enumerate(rat):
                    if r == 1:
                        if ev_st == -1:
                            ev_st = iii
                    else:
                        if ev_st != -1:
                            h.append({"start_token": ev_st, "end_token": iii})
                            ev_st = -1
                if ev_st != -1:
                    h.append({"start_token": ev_st, "end_token": iii + 1})

                l2.append(d)
            l.append(l2)
        return l

    def update_logits_rational(self, t_: tensor, meta: MetadataField, method=None) -> Tensor:
        mid_token_pos = [m['mid_token_pos'] for m in meta]
        token_size = [m['token_size'] for m in meta]
        t = t_.detach().cpu()
        for i, r in enumerate(t):
            t[i].apply_(lambda x: x if x > r[mid_token_pos[i]]
                        else r[mid_token_pos[i]])
            t[i] = (t[i] - min(t[i])) / (max(t[i]) - min(t[i]) + 0.00000000001)
            t[i][token_size[i]:] = -1
        return t.to(t_.device)

    def encode(self, sent_query: TextField, use_last:bool=True):
        embedded_sent = self.embedder(sent_query)
        batch_size, num_tokens_per_sent, num_dim = embedded_sent.size()
        embedded_sent = embedded_sent[:, 0, :]

        if isinstance(self.encoder, torch.nn.Linear) == True or isinstance(self.encoder, FeedForward) == True:
            embedded_sent = self.encoder(embedded_sent)
        elif isinstance(self.encoder, torch.nn.LSTM) == True:
            embedded_sent = embedded_sent.reshape((batch_size, 1, num_dim))
            embedded_sent, _ = self.encoder(embedded_sent)
        embedded_sent = self.dropout(embedded_sent)

        return embedded_sent, batch_size, num_tokens_per_sent, num_dim

    def forward(self, sent_query: TextField, evidences: TextField = None,
                label_target: LabelField = None,
                meta: MetadataField = None) -> Dict[str, torch.Tensor]:
        evidences = evidences.double()

        embedded_sent, batch_size, num_tokens_per_sent, num_dim = self.encode(sent_query=sent_query)

        logits = self.classifier(embedded_sent).squeeze().view(batch_size, -1)
        probs = F.softmax(logits, dim=1)

        logits_rational = self.rational_mclassifier(
            embedded_sent).squeeze().view(batch_size, -1)
        logits_rational = logits_rational[:, :num_tokens_per_sent]
        logits_rational = self.update_logits_rational(
            logits_rational, meta, self.rational_cast_method)

        output_dict = {}

        if True:
            labels = meta[0]['labels']
            classification_scores = []
            for l in probs:
                c = {}
                for i, _ in enumerate(labels):
                    c[labels[i]] = l[i]
                classification_scores.append(c)

            output_dict = {
                "rationales": self.get_evidences_dict(logits_rational, meta),
                "annotation_id": [m['annotation_id'] for m in meta],
                "classification": [labels[m] for m in torch.argmax(logits, axis=1)],
                "classification_scores": classification_scores,
                # "rat_val": logits_rational
            }

        if label_target is not None:
            loss1 = self.loss_fn_target(logits, label_target)
            loss2 = self.loss_fn_rational(logits_rational,
                                          evidences)

            output_dict['loss'] = self.loss_co1 * \
                loss1 + self.loss_co2 * loss2 + self.loss_b
            self.acc(probs, label_target)
            self.acc_rational(self.complete_tensor(logits_rational),
                              self.complete_tensor(evidences),
                              num_tokens_per_sent,
                              self.gen_mask(evidences))

        return output_dict


@Model.register('fine_tune_baseline_rational_to_pred_sp')
class FineTuneBaselineRationalToPredictSoft(FineTuneBaseline):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 serialization_dir: Optional[str],
                 regularizer: Optional[RegularizerApplicator] = None,
                 target_class_num: int = 3,
                 encoder:Union[StackedBidirectionalLstm,
                                   FeedForward] = None,
                 classifier: FeedForward = None,
                 rational_mclassifier: FeedForward = None,
                 dropout: float = 0.0,
                 rational_cast_method: str = RATIONAL_CAST_CUT,
                 loss_co1: int = 1,
                 loss_co2: int = 1,
                 loss_co3: int = 1,
                 loss_b: int = 0) -> None:
        super().__init__(vocab, embedder, serialization_dir, regularizer=regularizer,
                         target_class_num=target_class_num, classifier=classifier,
                         encoder=encoder,
                         rational_mclassifier=rational_mclassifier, dropout=dropout,
                         loss_co1=loss_co1, loss_co2=loss_co2,
                         loss_co3=loss_co3, loss_b=loss_b)
        if classifier is not None:
            self.classifier = FeedForward(
                input_dim=self.embedder.get_output_dim(),
                hidden_dims=self.classifier.get_output_dim(),
                num_layers=len(list(self.classifier._linear_layers)),
                activations=torch.nn.ReLU())


    def forward(self, sent_query: TextField, evidences: TextField = None,
                label_target: LabelField = None,
                meta: MetadataField = None) -> Dict[str, torch.Tensor]:
        evidences = evidences.double()

        embedded_sent, batch_size, num_tokens_per_sent, num_dim = self.encode(sent_query=sent_query)

        base_logits_rational = self.rational_mclassifier(
            embedded_sent).squeeze().view(batch_size, -1)
        logits_rational = base_logits_rational[:, :num_tokens_per_sent]
        logits_rational = self.update_logits_rational(
            logits_rational, meta, self.rational_cast_method)

        logits = self.classifier(
            base_logits_rational).squeeze().view(batch_size, -1)
        probs = F.softmax(logits, dim=1)

        output_dict = {}

        if True:
            labels = meta[0]['labels']
            classification_scores = []
            for l in probs:
                c = {}
                for i, _ in enumerate(labels):
                    c[labels[i]] = l[i]
                classification_scores.append(c)

            output_dict = {
                "rationales": self.get_evidences_dict(logits_rational, meta),
                "annotation_id": [m['annotation_id'] for m in meta],
                "classification": [labels[m] for m in torch.argmax(logits, axis=1)],
                "classification_scores": classification_scores,
                # "rat_val": logits_rational
            }

        if label_target is not None:
            loss1 = self.loss_fn_target(logits, label_target)
            loss2 = self.loss_fn_rational(logits_rational,
                                          evidences)

            output_dict['loss'] = self.loss_co1 * \
                loss1 + self.loss_co2 * loss2 + self.loss_b
            self.acc(probs, label_target)
            self.acc_rational(self.complete_tensor(logits_rational),
                              self.complete_tensor(evidences),
                              num_tokens_per_sent,
                              self.gen_mask(evidences))

        return output_dict


@Model.register('fine_tune_baseline_rational_cnt_to_pred_sp')
class FineTuneBaselineRationalToPredict(FineTuneBaseline):
    def __init__(self, vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 serialization_dir: Optional[str],
                 regularizer: Optional[RegularizerApplicator] = None,
                 target_class_num: int = 3,
                 classifier: Union[StackedBidirectionalLstm,
                                   FeedForward] = None,
                 rational_mclassifier: FeedForward = None,
                 dropout: float = 0.0,
                 rational_cnt_reg: FeedForward = None,
                 loss_co1: int = 1,
                 loss_co2: int = 1,
                 loss_co3: int = 1,
                 loss_b: int = 0) -> None:
        super().__init__(vocab, embedder, serialization_dir, regularizer=regularizer,
                         target_class_num=target_class_num, classifier=classifier,
                         rational_mclassifier=rational_mclassifier, dropout=dropout,
                         loss_co1=loss_co1, loss_co2=loss_co2,
                         loss_co3=loss_co3, loss_b=loss_b)

        self.embedder = embedder
        self.classifier = classifier or torch.nn.Linear(
            embedder.get_output_dim() + 1, target_class_num)
        self.rational_mclassifier = rational_mclassifier or torch.nn.Linear(
            embedder.get_output_dim(), embedder.get_output_dim())
        self.rational_cnt_reg = rational_cnt_reg or torch.nn.Linear(
            embedder.get_output_dim(), 1)
        self.dropout = torch.nn.Dropout(dropout)

        self.loss_fn_target = torch.nn.CrossEntropyLoss()
        self.loss_fn_rational = torch.nn.BCEWithLogitsLoss()
        self.loss_fn_rational_cnt = torch.nn.MSELoss()

        self.acc = CategoricalAccuracy()
        self.acc_rational = RationalMetricSingleSent(
            num_classes=embedder.get_output_dim())
        self.acc_rational_num = RationalCntMetricSingleSent()
        self.num_rational_tokens = embedder.get_output_dim()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        m = self.acc_rational.get_metric(reset)
        c = self.acc_rational_num.get_metric(reset)
        return {
            'accuracy': self.acc.get_metric(reset),
            'rational_precision': m['precision'],
            'rational_recall': m['recall'],
            'rational_fscore': m['fscore'],
            'rational_cnt_mse': c['mse'],
        }

    def forward(self, sent_query: TextField, evidences: TextField = None,
                label_target: LabelField = None,
                meta: MetadataField = None) -> Dict[str, torch.Tensor]:
        evidences = evidences.double()

        embedded_sent, batch_size, num_tokens_per_sent, num_dim = self.encode(sent_query=sent_query)

        base_logits_rational = self.rational_mclassifier(
            embedded_sent).squeeze().view(batch_size, -1)
        logits_rational = base_logits_rational[:, :num_tokens_per_sent]
        logits_rational = self.update_logits_rational(
            logits_rational, meta, self.rational_cast_method)

        logits_rational_cnt = self.rational_cnt_reg(
            embedded_sent).squeeze().view(batch_size, -1)

        tmp_tensor = torch.cat((base_logits_rational, logits_rational_cnt), 1)
        logits = self.classifier(tmp_tensor).squeeze().view(batch_size, -1)
        probs = F.softmax(logits, dim=1)

        labels = meta[0]['labels']
        classification_scores = []
        for l in probs:
            c = {}
            for i, _ in enumerate(labels):
                c[labels[i]] = l[i]
            classification_scores.append(c)

        output_dict = {
            "rationales": self.get_evidences_dict(logits_rational, meta),
            "annotation_id": [m['annotation_id'] for m in meta],
            "classification": [labels[m] for m in torch.argmax(logits, axis=1)],
            "classification_scores": classification_scores,
        }

        if label_target is not None:
            loss1 = self.loss_fn_target(logits, label_target)
            loss2 = self.loss_fn_rational(logits_rational,
                                          evidences)
            evidence_cnt_target = [[m['evidence_cnt']] for m in meta]
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
            evidence_cnt_target = torch.FloatTensor(
                evidence_cnt_target).to(device)
            loss3 = self.loss_fn_rational_cnt(
                logits_rational_cnt, evidence_cnt_target)

            output_dict['loss'] = self.loss_co1 * loss1 + \
                self.loss_co2 * loss2 + self.loss_co3 * loss3 + self.loss_b
            self.acc(probs, label_target)
            self.acc_rational(self.complete_tensor(logits_rational),
                              self.complete_tensor(evidences),
                              num_tokens_per_sent,
                              self.gen_mask(evidences))
            self.acc_rational_num(logits_rational_cnt, evidence_cnt_target)

        return output_dict
