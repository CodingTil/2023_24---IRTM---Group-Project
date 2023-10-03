import models.base as base_module
import models.T5Rewriter as t5_rewriter

from typing import List, Tuple

import pandas as pd
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker


class Baseline(base_module.Pipeline):
    """
    A class to represent the baseline retrieval method.

    Attributes
    ----------
    stages : List[Tuple[pt.Transformer, int]]
        The stages of the pipeline.
    """

    stages: List[Tuple[pt.Transformer, int]]

    def __init__(
        self,
        index,
        bm25_docs: int = 1000,
        mono_t5_docs: int = 100,
        duo_t5_docs: int = 10,
    ):
        """
        Constructs all the necessary attributes for the baseline retrieval method.

        Parameters
        ----------
        index : pt.Index
            The PyTerrier index.
        bm25_docs : int
            The number of documents to retrieve with BM25.
        mono_t5_docs : int
            The number of documents to retrieve with MonoT5.
        duo_t5_docs : int
            The number of documents to retrieve with DuoT5.
        """
        t5_qr = t5_rewriter.T5Rewriter(index)
        bm25 = pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"])
        mono_t5 = MonoT5ReRanker()
        duo_t5 = DuoT5ReRanker()

        top_docs = t5_qr >> bm25

        self.stages = [
            (top_docs, bm25_docs),
            (mono_t5, mono_t5_docs),
            (duo_t5, duo_t5_docs),
        ]

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

    def transform(self, query_df: pd.DataFrame) -> pd.DataFrame:
        # We basically do the pyterrier Concatenate transformer operator here, but more efficiently, since we dont have to do the entire pipeline for each component of the operator.
        results = []
        current_df = query_df

        is_first: bool = True

        for stage, num_docs in self.stages:
            df = current_df
            if not is_first:
                df = df.groupby("qid").head(num_docs)
            else:
                is_first = False
            transformed_df = stage.transform(df)
            transformed_df = transformed_df.groupby("qid").head(num_docs)
            results.append(transformed_df)
            current_df = transformed_df

        return self.combine_result_stages(results)
