import logging
from typing import Dict, Tuple, List
import csv
import os

import pandas as pd

import indexer.index as index_module
import models.base as base_model
import models.model_parameters as model_parameters_module

index = None
pipeline: base_model.Pipeline


def to_trec_runfile_format(df: pd.DataFrame, model_name: str) -> str:
    """
    Convert a dataframe to the TREC runfile format.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to convert.
    model_name : str
        The name of the model.

    Returns
    -------
    str
        The dataframe in the TREC runfile format.
    """
    return "\n".join(
        [
            f"{row['qid']} Q0 {row['docno']} {int(row['rank']) + 1} {row['score']} {model_name}"
            for _, row in df.iterrows()
        ]
    )


def main(
    *,
    recreate: bool,
    queries_file_path: str,
    output_file_path: str,
    model_parameters: model_parameters_module.ParametersBase,
) -> None:
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
    model_parameters : model_parameters_module.ParametersBase
        The model parameters.
    """
    global index
    global pipeline

    index = index_module.get_index(recreate=recreate)
    pipeline = model_parameters.create_Pipeline(index=index)

    logging.info("Loading queries...")
    queries: Dict[int, Dict[int, base_model.Query]] = {}  # topic_id -> (turn_id, query)
    with open(queries_file_path, "r") as queries_file:
        # Skip the header
        queries_file.readline()
        csv_reader = csv.reader(queries_file)
        for line in csv_reader:
            query_id, query, topic_id, turn_id = tuple(line[0:4])
            queries.setdefault(int(topic_id), {})[int(turn_id)] = base_model.Query(
                query_id=query_id, query=query
            )
    inputs: List[Tuple[List[base_model.Query], base_model.Context]] = []
    for topic_id, qs in queries.items():
        inputs.append(
            ([query for _, query in sorted(qs.items(), key=lambda x: x[0])], [])
        )

    logging.info("Running queries...")
    _, results = pipeline.batch_search_conversation(inputs)

    logging.info("Writing results...")
    # create file and parent directories if not exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as output_file:
        output_file.write(to_trec_runfile_format(results, "baseline"))
