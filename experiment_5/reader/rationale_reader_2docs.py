import os
from typing import Dict, List, Tuple

from overrides import overrides
import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, MetadataField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

from eraserbenchmark.rationale_benchmark.utils import annotations_from_jsonl, load_flattened_documents, Evidence

COSE_DATASET = 'cose'
ETC_DATASET = 'etc'


@DatasetReader.register("rationale_reader_2docs")
class RationaleReader2Docs(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        max_sequence_length: int = None,
        keep_prob: float = 1.0,
        lazy: bool = False
    ) -> None:
        super().__init__()
        self._max_sequence_length = max_sequence_length
        self._token_indexers = token_indexers

        self._keep_prob = keep_prob
        self._bert = "bert" in token_indexers
        self.ds_type = ETC_DATASET

    def generate_document_evidence_map(self, evidences: List[List[Evidence]]) -> Dict[str, List[Tuple[int, int]]]:
        document_evidence_map = {}
        for evgroup in evidences:
            for evclause in evgroup:
                if evclause.docid not in document_evidence_map:
                    document_evidence_map[evclause.docid] = []
                document_evidence_map[evclause.docid].append(
                    (evclause.start_token, evclause.end_token))

        return document_evidence_map

    @overrides
    def _read(self, file_path):
        data_dir = os.path.dirname(file_path)
        annotations = annotations_from_jsonl(file_path)
        documents: Dict[str, List[str]] = load_flattened_documents(
            data_dir, docids=None)

        # update ds_type
        if COSE_DATASET in file_path.lower():
            self.ds_type = COSE_DATASET

        for _, line in enumerate(annotations):
            annotation_id: str = line.annotation_id
            evidences: List[List[Evidence]] = line.evidences
            label: str = line.classification
            query: str = line.query
            docids: List[str] = sorted(
                list(set([evclause.docid for evgroup in evidences for evclause in evgroup])))

            filtered_documents: Dict[str, List[str]] = dict(
                [(d, documents[d]) for d in docids])
            document_evidence_map = self.generate_document_evidence_map(
                evidences)

            if label is not None:
                label = str(label)

            instance = self.text_to_instance(
                annotation_id=annotation_id,
                documents=filtered_documents,
                rationales=document_evidence_map,
                query=query,
                label=label,
            )
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(
        self,
        annotation_id: str,
        documents: Dict[str, List[str]],
        rationales: Dict[str, List[Tuple[int, int]]],
        query: str = None,
        label: str = None,
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}

        tokens = []
        premise_tokens = []
        query_tokens = []
        is_evidence = []

        document_to_span_map = {}
        always_keep_mask = []
        premise_always_keep_mask = []
        query_always_keep_mask = []

        for docid, docwords in documents.items():
            document_tokens = [Token(word) for word in docwords]
            tokens += document_tokens
            premise_tokens += document_tokens
            document_to_span_map[docid] = (
                len(tokens) - len(docwords), len(tokens))

            always_keep_mask += [0] * len(document_tokens)
            premise_always_keep_mask += [0] * len(document_tokens)

            tokens.append(Token("[SEP]"))
            premise_tokens.append(Token("[SEP]"))

            always_keep_mask += [1]
            premise_always_keep_mask += [1]

            rationale = [0] * len(docwords)
            if docid in rationales:
                for s, e in rationales[docid]:
                    for i in range(s, e):
                        rationale[i] = 1

            is_evidence += rationale + [1]

        if query is not None and type(query) != list:
            query_words = query.split()
            tokens += [Token(word) for word in query_words]
            query_tokens += [Token(word) for word in query_words]
            tokens.append(Token("[SEP]"))
            query_tokens.append(Token("[SEP]"))

            is_evidence += [1] * (len(query_words) + 1)
            always_keep_mask += [1] * (len(query_words) + 1)
            query_always_keep_mask += [1] * (len(query_words) + 1)
        elif query is None:
            query_tokens.append(Token("[SEP]"))
            query_always_keep_mask += [1]

        fields["document"] = TextField(tokens, self._token_indexers)
        fields["premise"] = TextField(premise_tokens, self._token_indexers)
        fields["query"] = TextField(query_tokens, self._token_indexers)

        fields["kept_tokens"] = SequenceLabelField(
            always_keep_mask, sequence_field=fields["document"], label_namespace="kept_token_labels"
        )
        fields["premise_kept_tokens"] = SequenceLabelField(
            premise_always_keep_mask, sequence_field=fields[
                "premise"], label_namespace="premise_kept_token_labels"
        )
        fields["query_kept_tokens"] = SequenceLabelField(
            query_always_keep_mask, sequence_field=fields["query"], label_namespace="query_kept_token_labels"
        )

        fields["rationale"] = SequenceLabelField(
            is_evidence, sequence_field=fields["document"], label_namespace="evidence_labels"
        )

        metadata = {
            "annotation_id": annotation_id,
            "tokens": tokens,
            "premise_tokens": premise_tokens,
            "query_tokens": query_tokens,
            "document_to_span_map": document_to_span_map,
            "convert_tokens_to_instance": self.convert_tokens_to_instance,
            "always_keep_mask": np.array(always_keep_mask)
        }

        fields["metadata"] = MetadataField(metadata)

        if label is not None:
            if self.ds_type == COSE_DATASET:
                query = query.split("[sep]")
                query = [x.strip() for x in query]
                fields["label"] = MetadataField({k: v for k, v in zip(
                    ["A", "B", "C", "D", "E", "Label"], query + [label])})
            else:
                fields["label"] = LabelField(label, label_namespace="labels")

        return Instance(fields)

    # def convert_tokens_to_instance(self, tokens, labels=None):
    #     return [Instance({"document": TextField(tokens, self._token_indexers)})]

    def convert_tokens_to_instance(self, tokens, labels:List[str] = None):
        return [Instance({"document": TextField(tokens + [Token(l)], self._token_indexers)}) for l in labels]
