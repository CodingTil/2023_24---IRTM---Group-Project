import models.base as base_module
import models.T5Rewriter as t5_rewriter

from typing import List, Tuple

import pandas as pd
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker

import torch

BATCH_SIZE = 128 if torch.cuda.is_available() else 8


class Baseline(base_module.Pipeline):
    """
    A class to represent the baseline retrieval method.
    """

    top_docs: Tuple[pt.Transformer, int]
    mono_t5: Tuple[MonoT5ReRanker, int]
    duo_t5: Tuple[DuoT5ReRanker, int]

    def __init__(
        self,
        index,
        *,
        bm25_docs: int,
        mono_t5_docs: int,
        duo_t5_docs: int,
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
        t5_qr = t5_rewriter.T5Rewriter()
        bm25 = pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"])
        self.top_docs = (t5_qr >> bm25, bm25_docs)
        self.mono_t5 = (MonoT5ReRanker(batch_size=BATCH_SIZE), mono_t5_docs)
        self.duo_t5 = (DuoT5ReRanker(batch_size=BATCH_SIZE), duo_t5_docs)

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
        top_docs_df = self.top_docs[0].transform(query_df)
        top_docs_df = (
            top_docs_df.sort_values(["qid", "score"], ascending=False)
            .groupby("qid")
            .head(self.top_docs[1])
        )

        mono_t5_df = self.mono_t5[0].transform(
            top_docs_df.groupby("qid").head(self.mono_t5[1])
        )
        mono_t5_df = (
            mono_t5_df.sort_values(["qid", "score"], ascending=False)
            .groupby("qid")
            .head(self.mono_t5[1])
        )

        duo_t5_df = self.duo_t5[0].transform(
            mono_t5_df.groupby("qid").head(self.duo_t5[1])
        )
        duo_t5_df = (
            duo_t5_df.sort_values(["qid", "score"], ascending=False)
            .groupby("qid")
            .head(self.duo_t5[1])
        )

        return self.combine_result_stages([top_docs_df, mono_t5_df, duo_t5_df])
