from typing import Optional, Dict, Any
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from allennlp.training.metrics import FBetaMeasure, CategoricalAccuracy


class BaseModel(Model):
    LOSS_MODE_ALL = 'all'
    LOSS_MODE_RATIONALE_ONLY = 'rationale_only'
    LOSS_MODE_OBJECTIVE_ONLY = 'objective_only'

    def __init__(
        self,
        vocab: Vocabulary,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        loss_mode: str=LOSS_MODE_ALL
    ):
        super(BaseModel, self).__init__(vocab, regularizer)
        self._vocabulary = vocab
        self._f1_metric = FBetaMeasure()
        self._accuracy = CategoricalAccuracy()

        self.prediction_mode = False
        self.loss_mode = loss_mode

        initializer(self)

    def forward(self, document, sentence_indices, query=None, labels=None, metadata=None):
        raise NotImplementedError

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        output_dict = self._decode(output_dict)
        return output_dict

    def _call_metrics(self, output_dict):
        self._f1_metric(output_dict['logits'], output_dict['gold_labels'])
        self._accuracy(output_dict['logits'], output_dict['gold_labels'])

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self._f1_metric.get_metric(reset)
        output_labels = self._vocabulary.get_index_to_token_vocabulary(
            "labels")
        if len(output_labels) == 0:  # COSE
            output_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        output_labels = [output_labels[i] for i in range(len(output_labels))]

        class_metrics = {}
        for k, v in metrics.items():
            assert len(v) == len(output_labels)
            class_nums = dict(zip(output_labels, v))
            class_metrics.update(
                {k + "_" + str(kc): x for kc, x in class_nums.items()})

        class_metrics.update({"accuracy": self._accuracy.get_metric(reset)})
        modified_class_metrics = {}

        for k, v in class_metrics.items():
            if k.endswith('_1') or k == 'accuracy':
                modified_class_metrics[k] = v
            else:
                modified_class_metrics['_' + k] = v

        return modified_class_metrics

    def normalize_attentions(self, output_dict):
        '''
        In case, attention is over subtokens rather than at token level. 
        Combine subtoken attention into token attention.
        '''

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
