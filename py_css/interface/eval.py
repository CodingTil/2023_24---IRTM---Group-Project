import logging

import pyterrier as pt

import indexer.index as index_module
import models.baseline as baseline_module

index = None
pipeline: pt.Transformer


def main(*, recreate: bool, queries_file_path: str, qrels_file_path: str) -> None:
    """
    The main function of the eval interface.

    Parameters
    ----------
    recreate : bool
        Whether to recreate the index.
    queries_file_path : str
        The path to the queries file.
    qrels_file_path : str
        The path to the qrels file.
    """
    global index
    global pipeline

    index = index_module.get_index(recreate=recreate)
    pipeline = baseline_module.create_pipeline(index)
