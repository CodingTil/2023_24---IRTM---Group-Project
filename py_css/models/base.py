from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Optional, List, Tuple, TypeAlias, Set, Any, Dict, Generator
import warnings

import models.T5Rewriter as t5_rewriter_module

import pandas as pd


@dataclass
class Query:
    """
    A class to represent a query.

    Attributes
    ----------
    query_id : str
        The id of the query (topicid_turnid)
    query : str
        The text of the query.
    """

    query_id: str
    query: str

    def get_topic_id(self) -> int:
        return int(self.query_id.split("_")[0])

    def get_turn_id(self) -> int:
        return int(self.query_id.split("_")[1])

    def __str__(self) -> str:
        return self.query


@dataclass
class Document:
    """
    A class to represent a document.

    Attributes
    ----------
    docno : str
        The id of the document.
    content : str
        The content of the document.
    """

    docno: str
    content: str

    def __str__(self) -> str:
        return self.content


Context: TypeAlias = List[Tuple[Query, Optional[List[Document]]]]

# If the retrieval model did not find any suitable or all N-required documents, this document shall be used as a placeholder
EMPTY_PLACEHOLDER_DOC: Document = Document("-1", "")


class Pipeline(ABC):
    """
    Abstract base class for all pipelines.
    """

    @abstractmethod
    def transform_input(self, query: Query, context: Context) -> str:
        """
        Transform the input query.

        Parameters
        ----------
        query : Query
            The query to be transformed.
        context : Context
            The context of the query.

        Returns
        -------
        str
            The transformed query.
        """
        ...

    @abstractmethod
    def transform(self, query_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input queries.

        Parameters
        ----------
        query_df : pd.DataFrame
            The queries to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed queries.
        """
        ...

    def pad_empty_documents(
        self, df: pd.DataFrame, qids: Set[str], N: int, queries_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Pad the dataframe with empty documents.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be padded.
        qids : Set[str]
            The query ids.
        N : int
            The number of documents each qid should have.
        queries_df : pd.DataFrame
            The queries dataframe.

        Returns
        -------
        pd.DataFrame
            The padded dataframe.
        """
        df_has_rewritten_queries: bool = (
            t5_rewriter_module.COPY_REWRITTEN_QUERY_COLUMN in df.columns
        )
        rows_to_add: List[Dict[str, Any]] = []
        for qid in qids:
            # Check if qid is in top_doc_df (if not --> no documents at all were found)
            if qid not in df["qid"].unique():
                for i in range(1, N + 1):
                    row = {
                        "qid": qid,
                        "docid": EMPTY_PLACEHOLDER_DOC.docno,
                        "docno": EMPTY_PLACEHOLDER_DOC.docno,
                        "text": EMPTY_PLACEHOLDER_DOC.content,
                        "score": -i,
                        "rank": i,
                        "query_0": queries_df[queries_df["qid"] == qid]["query"].iloc[
                            0
                        ],
                        "query": queries_df[queries_df["qid"] == qid]["query"].iloc[0],
                    }
                    if df_has_rewritten_queries:
                        row[t5_rewriter_module.COPY_REWRITTEN_QUERY_COLUMN] = row[
                            "query"
                        ]
                    rows_to_add.append(row)
            else:
                # if there are less than N occurrences of a qid, add base_module.EMPTY_PLACEHOLDER_DOC to fill up (need to adjust score)
                if df.groupby("qid").size()[qid] < N:
                    lowest_score = df[df["qid"] == qid]["score"].min()
                    rank = int(round(df[df["qid"] == qid]["rank"].max()))
                    for i in range(
                        rank + 1,
                        N + 1,
                    ):
                        row = {
                            "qid": qid,
                            "docid": EMPTY_PLACEHOLDER_DOC.docno,
                            "docno": EMPTY_PLACEHOLDER_DOC.docno,
                            "text": EMPTY_PLACEHOLDER_DOC.content,
                            "score": lowest_score - i,
                            "rank": i,
                            "query_0": queries_df[queries_df["qid"] == qid][
                                "query"
                            ].iloc[0],
                            "query": queries_df[queries_df["qid"] == qid]["query"].iloc[
                                0
                            ],
                        }
                        if df_has_rewritten_queries:
                            row[t5_rewriter_module.COPY_REWRITTEN_QUERY_COLUMN] = row[
                                "query"
                            ]
                        rows_to_add.append(row)
        if len(rows_to_add) > 0:
            df = pd.concat([df, pd.DataFrame(rows_to_add)])
            df = df.sort_values(["qid", "rank"], ascending=[True, True])

        return df

    def replace_empty_placeholder_docs(
        self, df: pd.DataFrame, context_list: List[Tuple[Query, Context]]
    ) -> pd.DataFrame:
        """
        Replace any empty placeholder documents with documents from the context, if possible.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be replaced.
        context_list : List[Tuple[Query, Context]]
            The context of the queries.

        Returns
        -------
        pd.DataFrame
            The dataframe with the replaced documents.
        """

        def gen_context_docs(context: Context) -> Generator[Document, None, None]:
            for _, docs in reversed(context):
                if docs is not None:
                    for doc in docs:
                        if doc.docno != EMPTY_PLACEHOLDER_DOC.docno:
                            yield doc

        # if in df there are multiple rows that have the same qid and docno, keep the one with the highest score. For the ones removed, add a row each with the EMPTY_PLACEHOLDER_DOC
        rank_size_per_qid: int = df.groupby("qid").size().max()
        print(f"Rank size per qid: {rank_size_per_qid}")
        df = df.sort_values(["qid", "docno", "score"], ascending=[True, True, False])
        total_size = df.shape[0]
        df = df.drop_duplicates(subset=["qid", "docno"], keep="first")
        dropped_any: bool = total_size != df.shape[0]
        print(f"Dropped any: {dropped_any}")
        df = df.reset_index(drop=True)
        df = self.pad_empty_documents(
            df, df["qid"].unique(), rank_size_per_qid, df[["qid", "query"]]
        )
        print(f"Number of max rank size per qid: {df.groupby('qid').size().max()}")
        df = df.reset_index(drop=True)
        df = df.sort_values(["qid", "rank"], ascending=[True, True])

        for query, context in context_list:
            # check if there is a row in the df with "qid" == query.query_id, where "docno" == EMPTY_PLACEHOLDER_DOC.docno
            # if yes, replace it with the top document from the context
            while True:
                if not df[
                    (df["qid"] == query.query_id)
                    & (df["docno"] == EMPTY_PLACEHOLDER_DOC.docno)
                ].empty:
                    # Check if gen_docs has next element
                    doc: Document
                    doc_gen = gen_context_docs(context)
                    try:
                        doc = next(doc_gen)
                        while (
                            doc.docno
                            in df[(df["qid"] == query.query_id)]["docno"].unique()
                        ):
                            doc = next(doc_gen)
                    except StopIteration:
                        break
                    # Get the row index of the row to be replaced (of all of the rows satisfying the condition, take the one with min "rank" value)
                    row_index = df[
                        (df["qid"] == query.query_id)
                        & (df["docno"] == EMPTY_PLACEHOLDER_DOC.docno)
                    ]["rank"].idxmin()

                    # Of that row, set docno and docid to doc.no, and text to doc.content
                    df.loc[row_index, "docno"] = doc.docno
                    df.loc[row_index, "docid"] = doc.docno
                    df.loc[row_index, "text"] = doc.content
                else:
                    break

        return df

    def combine_result_stages(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine the results of the stages.

        Parameters
        ----------
        results : List[pd.DataFrame]
            The results of the stages.

        Returns
        -------
        pd.DataFrame
            The combined results.
        """
        epsilon = 0.0001

        # Start with the rightmost dataframe as the base
        combined_res = results[-1].copy()

        for res in reversed(
            results[:-1]
        ):  # For each remaining dataframe from right to left
            # Identify the lowest score for each query in combined_res
            last_scores = (
                combined_res[["qid", "score"]]
                .groupby("qid")
                .min()
                .rename(columns={"score": "_lastscore"})
            )

            # Find the intersection of documents between current res and combined_res
            intersection = pd.merge(
                combined_res[["qid", "docno"]], res[["qid", "docno"]].reset_index()
            )
            remainder = res.drop(intersection["index"])

            # Identify the highest score for each query in remainder
            first_scores = (
                remainder[["qid", "score"]]
                .groupby("qid")
                .max()
                .rename(columns={"score": "_firstscore"})
            )

            # Adjust the scores in remainder to ensure proper ordering
            remainder = remainder.merge(last_scores, on=["qid"]).merge(
                first_scores, on=["qid"]
            )
            remainder["score"] = (
                remainder["score"]
                - remainder["_firstscore"]
                + remainder["_lastscore"]
                - epsilon
            )
            remainder = remainder.drop(columns=["_lastscore", "_firstscore"])

            # Concatenate remainder with combined_res and sort
            combined_res = pd.concat([combined_res, remainder]).sort_values(
                by=["qid", "score", "docno"], ascending=[True, False, True]
            )

        # Recompute the ranks
        combined_res = combined_res.assign(
            rank=combined_res.groupby("qid")["score"].rank(
                method="first", ascending=False
            )
        )

        return combined_res

    def search(self, query: Query, context: Context) -> Tuple[Context, pd.DataFrame]:
        """
        Search for the query.

        Parameters
        ----------
        query : Query
            The query to be searched.
        context : Context
            The context of the query.

        Returns
        -------
        Tuple[Context, pd.DataFrame]
            The updated context and the result of the search.
        """
        query_str = self.transform_input(query, context)
        query_df = pd.DataFrame([{"qid": query.query_id, "query": query_str}])
        result = self.transform(query_df)
        result = self.replace_empty_placeholder_docs(result, [(query, context)])

        if t5_rewriter_module.COPY_REWRITTEN_QUERY_COLUMN in result.columns:
            temp_result = result[result["qid"] == query.query_id]
            if not temp_result.empty:
                query.query = temp_result[
                    t5_rewriter_module.COPY_REWRITTEN_QUERY_COLUMN
                ].iloc[0]
            else:
                warnings.warn(
                    f"Query {query.query_id} not found in result. This should not happen. All query-ids: {result['qid'].unique()}"
                )
        else:
            temp_result = result[result["qid"] == query.query_id]
            if not temp_result.empty:
                query.query = temp_result["query"].iloc[0]
            else:
                warnings.warn(
                    f"Query {query.query_id} not found in result. This should not happen. All query-ids: {result['qid'].unique()}"
                )

        doc_list: List[Document] = []
        for _, entry in result.iterrows():
            doc_list.append(Document(entry["docno"], entry["text"]))

        context.append((query, doc_list))

        return context, result

    def batch_search(
        self, inputs: List[Tuple[Query, Context]]
    ) -> Tuple[List[Context], pd.DataFrame]:
        """
        Batch search for the queries.

        Parameters
        ----------
        inputs : List[Tuple[Query, Context]]
            The queries to be searched.

        Returns
        -------
        Tuple[List[Context], pd.DataFrame]
            The updated contexts and the result of the search.
        """
        query_df = pd.DataFrame(
            [
                {"qid": q.query_id, "query": self.transform_input(q, c)}
                for q, c in inputs
            ]
        )
        result = self.transform(query_df)
        result = self.replace_empty_placeholder_docs(result, inputs)

        if t5_rewriter_module.COPY_REWRITTEN_QUERY_COLUMN in result.columns:
            for query, _ in inputs:
                temp_result = result[result["qid"] == query.query_id]
                if not temp_result.empty:
                    query.query = temp_result[
                        t5_rewriter_module.COPY_REWRITTEN_QUERY_COLUMN
                    ].iloc[0]
                else:
                    warnings.warn(
                        f"Query {query.query_id} not found in result. This should not happen. All query-ids: {result['qid'].unique()}"
                    )
        else:
            for query, _ in inputs:
                temp_result = result[result["qid"] == query.query_id]
                if not temp_result.empty:
                    query.query = temp_result["query"].iloc[0]
                else:
                    warnings.warn(
                        f"Query {query.query_id} not found in result. This should not happen. All query-ids: {result['qid'].unique()}"
                    )

        contexts: List[Context] = []
        for query, context in inputs:
            query_id = query.query_id
            doc_list: List[Document] = []
            for _, entry in result[result["qid"] == query_id].iterrows():
                doc_list.append(Document(entry["docno"], entry["text"]))
            context.append((query, doc_list))
            contexts.append(context)

        return contexts, result

    def search_conversation(
        self, queries: List[Query], context: Context
    ) -> Tuple[Context, pd.DataFrame]:
        """
        Search for the queries in a conversation.

        Parameters
        ----------
        queries : List[Query]
            The queries to be searched.
        context : Context
            The context of the queries.

        Returns
        -------
        Tuple[Context, pd.DataFrame]
            The updated context and the result of the search.
        """
        results: List[pd.DataFrame] = []
        for query in queries:
            context, result = self.search(query, context)
            results.append(result)
        return context, pd.concat(results)

    def batch_search_conversation(
        self, inputs: List[Tuple[List[Query], Context]]
    ) -> Tuple[List[Context], pd.DataFrame]:
        """
        Batch search for the queries in a conversation.

        Parameters
        ----------
        inputs : List[Tuple[List[Query], Context]]
            The queries to be searched.

        Returns
        -------
        Tuple[List[Context], pd.DataFrame]
            The updated contexts and the result of the search.
        """
        results: List[pd.DataFrame] = []
        all_contexts: List[Context] = [context for _, context in inputs]

        max_len = max(
            len(queries) for queries, _ in inputs
        )  # Find the max number of queries in an input

        for idx in range(max_len):  # Loop through each position in the query list
            # Prepare the batch of queries that exist at position idx for each input
            current_batch = []
            ids = []
            for i, (queries, _) in enumerate(inputs):
                if idx < len(queries):
                    current_batch.append((queries[idx], all_contexts[i]))
                    ids.append(i)

            # Filter out None queries (inputs that don't have a query at position idx)
            valid_batch = [
                (query, context)
                for query, context in current_batch
                if query is not None
            ]

            # Batch search the queries
            updated_contexts, result = self.batch_search(valid_batch)

            # Update the results and contexts
            for i, context in zip(ids, updated_contexts):
                all_contexts[i] = context
            results.append(result)

            logging.info(f"Processed {idx+1}/{max_len} positions")

        return all_contexts, pd.concat(results)
