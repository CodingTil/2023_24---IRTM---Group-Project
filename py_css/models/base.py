from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Optional, List, Tuple, TypeAlias

import pandas as pd
import pyterrier as pt


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


class Pipeline(ABC):
    """
    Abstract base class for all pipelines.
    """

    pipeline: pt.Transformer

    def __init__(self, pipeline: pt.Transformer):
        self.pipeline = pipeline

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
        result = self.pipeline.search(query_str)

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
            [{"qid": q.query_id, "query": q.query} for q, _ in inputs]
        )
        result = self.pipeline.transform(query_df)

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
