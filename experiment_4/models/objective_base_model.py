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


@Model.register("objective_base")
class ObjectiveBaseModel(BaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        doc_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: FeedForward,
        attention: Attention,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(ObjectiveBaseModel, self).__init__(
            vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._doc_field_embedder = doc_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._dropout = torch.nn.Dropout(p=dropout)

        self._attention = attention

        self._feedforward_encoder = feedforward_encoder
        self._classifier_input_dim = self._feedforward_encoder.get_output_dim()

        self._num_labels = self._vocabulary.get_vocab_size("labels")
        if self._num_labels == 0:
            self._num_labels = 5 # for COSE

        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, self._num_labels)

        self._vector = torch.nn.Parameter(torch.randn(
            (1, self._seq2seq_encoder.get_output_dim())))

        initializer(self)

    def forward(self, sample_z, label=None, metadata=None) -> Dict[str, Any]:
        # premise
        doc = self._regenerate_tokens(
            sample_z=sample_z, metadata=metadata)
        embedded_text = self._doc_field_embedder(doc)
        mask = util.get_text_field_mask(doc).float()
        embedded_text = self._dropout(
            self._seq2seq_encoder(embedded_text, mask=mask))
        attentions = self._attention(vector=self._vector,
                                             matrix=embedded_text,
                                             matrix_mask=mask)
        embedded_text = embedded_text * \
            attentions.unsqueeze(-1) * mask.unsqueeze(-1)

        embedded_vec = self._feedforward_encoder(embedded_text.sum(1))

        logits = self._classification_layer(embedded_vec)
        probs = torch.nn.Softmax(dim=-1)(logits)

        output_dict = {}

        if label is not None:
            if isinstance(label, list):
                output_labels = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                label = torch.LongTensor(
                    [output_labels[l["Label"]] for l in label]).to(logits.device)

            loss = F.cross_entropy(logits, label, reduction="none")
            output_dict["loss"] = loss

        output_dict["logits"] = logits
        output_dict["probs"] = probs
        output_dict["class_probs"] = probs.max(-1)[0]
        output_dict["predicted_labels"] = probs.argmax(-1)
        output_dict["gold_labels"] = label
        output_dict["metadata"] = metadata

        self._call_metrics(output_dict)

        return output_dict

    def _regenerate_tokens(self, metadata, sample_z):
        sample_z_cpu = sample_z.cpu().data.numpy()
        tokens = [m["tokens"] for m in metadata]

        assert len(tokens) == len(sample_z_cpu)
        assert max([len(x) for x in tokens]) == sample_z_cpu.shape[1]

        instances = []
        new_tokens = []
        for words, mask, meta in zip(tokens, sample_z_cpu, metadata):
            mask = mask[: len(words)]
            new_words = [w for i, (w, m) in enumerate(
                zip(words, mask)) if i == 0 or m == 1]

            new_tokens.append(new_words)
            meta["new_tokens"] = new_tokens
            instance = metadata[0]["convert_tokens_to_instance"](
                new_words, None)
            instances += instance

        batch = Batch(instances)
        batch.index_instances(self._vocabulary)
        padding_lengths = batch.get_padding_lengths()

        batch = batch.as_tensor_dict(padding_lengths)
        return {"bert": {k: v.to(sample_z.device) for k, v in batch["document"]["bert"].items()}}
