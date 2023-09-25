from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, TypeAlias


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

    @abstractmethod
    def __init__(self, index):
        """
        Parameters
        ----------
        index : pyterrier.index.Index
            The index to be used.
        """
        ...

    @abstractmethod
    def search(self, query: Query, context: Context) -> Context:
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
        Context
            The context of the query after searching.
        """
        ...
