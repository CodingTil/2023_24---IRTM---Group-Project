import models.base as base_module
import models.T5Rewriter as t5_rewriter

import pyterrier as pt

# from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker


class Baseline(base_module.Pipeline):
    pipeline: pt.Transformer

    def __init__(self, index):
        # tokenizer = pt.rewrite.tokenise()
        # sdm = pt.rewrite.SequentialDependence()
        t5_qr = t5_rewriter.T5Rewriter(index)
        bm25 = pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"])
        bo1 = pt.rewrite.Bo1QueryExpansion(index)
        pl2 = pt.BatchRetrieve(index, wmodel="PL2", metadata=["docno", "text"])
        # mono_t5 = MonoT5ReRanker()
        # duo_t5 = DuoT5ReRanker()
        pipeline = (
            # tokenizer
            # >> sdm
            t5_qr
            >> (bm25 % 1000).compile()
            >> bo1
            >> pl2
            # >> pt.text.get_text(index, "text")
            # >> (mono_t5 % 1000)
            # >> duo_t5
        )
        super().__init__(pipeline)

    def transform_input(
        self, query: base_module.Query, context: base_module.Context
    ) -> str:
        history = []
        for q, _ in context:
            history.append(q.query)
        if len(context) > 0:
            last_docs = context[-1][1]
            if last_docs is not None:
                history.append(last_docs[0].content)
        history.append(query.query)
        new_query = " <sep> ".join(history)
        return new_query
