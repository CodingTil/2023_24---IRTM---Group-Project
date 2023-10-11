from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import models.base as base_model
import models.baseline as baseline_model


class ParametersBase(ABC):
    """
    An abstract class to represent the parameters of a model.
    """

    @abstractmethod
    def create_Pipeline(self, index) -> base_model.Pipeline:
        """
        Creates a pipeline with the given index.

        Parameters
        ----------
        index : pt.Index
            The PyTerrier index.

        Returns
        -------
        base_model.Pipeline
            The pipeline.
        """
        ...

    @staticmethod
    @abstractmethod
    def from_tuple(tup: Tuple) -> ParametersBase:
        """
        Creates a ParametersBase object from a tuple.

        Parameters
        ----------
        tup : Tuple
            The tuple.

        Returns
        -------
        ParametersBase
            The ParametersBase object.
        """
        ...


@dataclass
class BaselineParameters(ParametersBase):
    """
    A class to represent the parameters of the baseline retrieval method.

    Attributes
    ----------
    bm25_docs : int
        The number of documents to retrieve with BM25.
    mono_t5_docs : int
        The number of documents to rerank with MonoT5.
    duo_t5_docs : int
        The number of documents to rerank with DuoT5.
    """

    bm_25_docs: int
    mono_t5_docs: int
    duo_t5_docs: int

    def create_Pipeline(self, index) -> base_model.Pipeline:
        """
        Creates the baseline pipeline with the given index.

        Parameters
        ----------
        index : pt.Index
            The PyTerrier index.

        Returns
        -------
        base_model.Pipeline (baseline_model.Baseline)
            The baseline pipeline.
        """
        return baseline_model.Baseline(
            index,
            bm25_docs=self.bm_25_docs,
            mono_t5_docs=self.mono_t5_docs,
            duo_t5_docs=self.duo_t5_docs,
        )

    @staticmethod
    def from_tuple(tup: Tuple) -> ParametersBase:
        """
        Creates a BaselineParameters object from a tuple.

        Parameters
        ----------
        tup : Tuple[int, int, int]
            The tuple (bm25_docs, mono_t5_docs, duo_t5_docs)

        Returns
        -------
        ParametersBase (BaselineParameters)
            The ParametersBase object.

        Raises
        ------
        AssertionError
            If the tuple does not have 3 elements or if any of the elements is not a positive integer.
        """
        assert len(tup) == 3, "The tuple must have 3 elements."
        for i in tup:
            assert isinstance(i, int), "All parameters must be integers."
            assert i > 0, "All parameters must be positive integers."
        return BaselineParameters(tup[0], tup[1], tup[2])
