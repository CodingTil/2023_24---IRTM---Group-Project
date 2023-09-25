import models.base as base_module
from typing import Tuple, List

import pandas as pd
import pyterrier as pt


class Baseline(base_module.Pipeline):
    pipeline: pt.Transformer

    def __init__(self, index):
        tokenizer = pt.rewrite.tokenise()
        sdm = pt.rewrite.SequentialDependence()
        bm25 = pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"])
        bo1 = pt.rewrite.Bo1QueryExpansion(index)
        pl2 = pt.BatchRetrieve(index, wmodel="PL2", metadata=["docno", "text"])
        self.pipeline = tokenizer >> sdm >> (bm25 % 1000).compile() >> bo1 >> pl2

    def search(
        self, query: base_module.Query, context: base_module.Context
    ) -> Tuple[base_module.Context, pd.DataFrame]:
        query_str = str(query)
        result = self.pipeline.search(query_str)

        doc_list: List[base_module.Document] = []
        for _, entry in result.iterrows():
            doc_list.append(base_module.Document(entry["docno"], entry["text"]))

        context.append((query, doc_list))

        return context, result
