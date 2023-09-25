import models.base as base_module

import pyterrier as pt


class Baseline(base_module.Pipeline):
    pipeline: pt.Transformer

    def __init__(self, index):
        tokenizer = pt.rewrite.tokenise()
        sdm = pt.rewrite.SequentialDependence()
        bm25 = pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"])
        bo1 = pt.rewrite.Bo1QueryExpansion(index)
        pl2 = pt.BatchRetrieve(index, wmodel="PL2", metadata=["docno", "text"])
        pipeline = tokenizer >> sdm >> (bm25 % 1000).compile() >> bo1 >> pl2
        super().__init__(pipeline)

    def transform_input(
        self, query: base_module.Query, context: base_module.Context
    ) -> str:
        return query.query
