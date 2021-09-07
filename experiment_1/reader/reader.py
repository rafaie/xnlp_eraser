from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.fields import Field, ListField, MultiLabelField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, token_indexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
import os
from eraserbenchmark.rationale_benchmark.utils import annotations_from_jsonl, load_flattened_documents, Evidence
from allennlp.common.file_utils import cached_path
import codecs
import re

from typing import Dict, Iterable, List, Tuple
import logging
import os
import json

from overrides.overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('single_doc_reader')
class SingleDocReader(DatasetReader):
    def __init__(self,
                 labels: List[str],
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = 512,
                 namespace: str = 'default',
                 evidences_lbl: str = "evidence",
                 non_evidences_lbl: str = "non_evidence",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_tokens = max_tokens
        self.namespace = namespace
        self.docs = None
        self.evidences_lbl = evidences_lbl
        self.non_evidences_lbl = non_evidences_lbl
        self.labels = labels
        if len(labels) > 10:
            self.labels = (''.join(self.labels)).split(',')
        self.lbl_dict = {}

        c = 0
        for l in self.labels:
            self.lbl_dict[l] = c
            c += 1

    def generate_document_evidence_map(self, evidences: List[List[Evidence]]) -> Dict[str, List[Tuple[int, int]]]:
        document_evidence_map = {}
        for evgroup in evidences:
            for evclause in evgroup:
                if evclause.docid not in document_evidence_map:
                    document_evidence_map[evclause.docid] = []
                document_evidence_map[evclause.docid].append(
                    (evclause.start_token, evclause.end_token))

        return document_evidence_map

    def _read(self, file_path: str) -> Iterable[Instance]:
        data_dir = os.path.dirname(file_path)
        annotations = annotations_from_jsonl(file_path)
        documents: Dict[str, List[str]] = load_flattened_documents(
            data_dir, docids=None)
        for _, line in enumerate(annotations):
            annotation_id: str = line.annotation_id
            evidences: List[List[Evidence]] = line.evidences
            label: str = line.classification
            # query: str = line.query
            docids: List[str] = sorted(list(set([evclause.docid for evgroup in evidences for evclause in evgroup])))

            docs: Dict[str, List[str]] = dict([(d, documents[d]) for d in docids])
            evidences_map = self.generate_document_evidence_map(evidences)

            if label is not None:
                label = str(label)

            yield self.text_to_instance(docs, annotation_id, label, evidences, evidences_map)

    def text_to_tokens_plus_evidences_lbl(self,
                                          doc: List[str],
                                          evidences: Dict
                                          ) -> Tuple[List[Token], List[str]]:
        txt_split = doc
        tokens, offsets = self.tokenizer.intra_word_tokenize(txt_split)
        rat = [self.non_evidences_lbl] * len(tokens)
        for ev in evidences:
            for i in range(ev[0], ev[1]):
                for of in range(offsets[i][0], offsets[i][1] + 1):
                    rat[of] = self.evidences_lbl

        return tokens, rat, offsets

    def text_to_instance(self, documents: Dict,
                         annotation_id=None, label=None,
                         evidences=None, evidences_map: Dict=None) -> Instance:
        fields: Dict[str, Field] = {}
        if label is not None:
            fields['label_target'] = LabelField(
                self.lbl_dict[label.lower().strip()], skip_indexing=True)

        tokens = []
        evidences_lbl = []
        evidence_cnt = 0
        doc1 = []
        doc2 = []
        if evidences is not None and len(evidences) > 0:
            prem_evidences = []
            hypo_evidences = []
            hypo_offsets = []
            prem_evidences = []

            cnt = 0
            for e in evidences:
                if cnt == 2:
                    break
                for d in e:
                    if d.docid == list(documents.keys())[0]:
                        k = list(documents.keys())[0]
                        hypo_evidences = evidences_map[k]
                        evidence_cnt += sum([e[1] - e[0] for e in evidences_map[k]])
                        doc1 = documents[k]
                        cnt += 1
                    elif d.docid == list(documents.keys())[1]:
                        k = list(documents.keys())[1]
                        prem_evidences = evidences_map[k]
                        evidence_cnt += sum([e[1] - e[0] for e in evidences_map[k]])
                        doc2 = documents[k]
                        cnt += 1

            hypo_tokens, hypo_evidences, hypo_offsets = self.text_to_tokens_plus_evidences_lbl(
                doc1, hypo_evidences)
            prem_tokens, prem_evidences, prem_offsets = self.text_to_tokens_plus_evidences_lbl(
                doc2, prem_evidences)

            tokens = prem_tokens[:-1] + \
                self.tokenizer.sequence_pair_mid_tokens[:1] + \
                hypo_tokens[1:]
            evidences_lbl = prem_evidences[:-1] + \
                [self.non_evidences_lbl] + hypo_evidences[1:]
        else:
            prem_tokens = self.tokenizer.tokenize(documents[0])
            hypo_tokens = self.tokenizer.tokenize(documents[1]) if len(documents) > 1 else self.tokenizer.tokenize('')
            tokens = prem_tokens[:-1] + \
                self.tokenizer.self.sequence_pair_mid_tokens() + \
                hypo_tokens[1:]

        if annotation_id is not None:
            fields['meta'] = MetadataField({
                'annotation_id': annotation_id,
                'target_label': label,
                'evidence_cnt': evidence_cnt,
                'hypo_offsets': hypo_offsets,
                'prem_offsets': prem_offsets
            })

        fields['sent_query'] = TextField(
            tokens=tokens, token_indexers=self.token_indexers)
        fields['evidences'] = ListField(
            [LabelField(ev) for ev in evidences_lbl])

        return Instance(fields)
