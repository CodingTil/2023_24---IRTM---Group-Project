import logging
from typing import Dict, List, Tuple
import csv
import tempfile
import subprocess

import pyterrier as pt

import indexer.index as index_module
import models.base as base_model
import models.baseline as baseline_module
import interface.run_queries as run_queries_module

index = None
pipeline: base_model.Pipeline


def main(
    *,
    recreate: bool,
    queries_file_path: str,
    qrels_file_path: str,
    baseline_params: Tuple[int, int, int],
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
    baseline_params : Tuple[int, int, int]
        The parameters for the baseline model.
    """
    global index
    global pipeline

    index = index_module.get_index(recreate=recreate)
    pipeline = baseline_module.Baseline(
        index, baseline_params[0], baseline_params[1], baseline_params[2]
    )

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

    logging.info("Loading qrels...")
    qrels: Dict[str, Dict[str, str]] = {}  # query_id, document_id, relevance
    with open(qrels_file_path, "r") as qrels_file:
        for line in qrels_file:
            t = tuple(line.split())
            if len(t) == 4:
                query_id, _, document_id, relevance = t
                qrels.setdefault(query_id, {})[document_id] = relevance
            elif len(t) == 3:  # second missing, ignore
                query_id, document_id, relevance = t
                qrels.setdefault(query_id, {})[document_id] = relevance
    tmp_qrels_file = tempfile.NamedTemporaryFile(delete=True)
    # write qrels to tmp_qrels_file
    with open(tmp_qrels_file.name, "w") as f:
        for query_id, docs in qrels.items():
            for document_id, relevance in docs.items():
                f.write(f"{query_id} 0 {document_id} {relevance}\n")

    logging.info("Running queries...")
    _, results = pipeline.batch_search_conversation(inputs)

    trec_runtime_str = run_queries_module.to_trec_runfile_format(results, "baseline")
    tmpfile = tempfile.NamedTemporaryFile(delete=True)
    # write trec_runtime_str to tempfile
    with open(tmpfile.name, "w") as f:
        f.write(trec_runtime_str)

    logging.info("Evaluating...")
    # ./trec_eval -c -m recall.1000 -m map -m recip_rank -m ndcg_cut.3 -l2 -M1000 qrels_train.txt {YOUR_TREC_RUNFILE}
    process = subprocess.Popen(
        [
            "trec_eval",
            "-c",
            "-m",
            "recall.1000",
            "-m",
            "map",
            "-m",
            "recip_rank",
            "-m",
            "ndcg_cut.3",
            "-l2",
            "-M1000",
            tmp_qrels_file.name,
            tmpfile.name,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    out = stdout.decode("utf-8")
    if len(out) > 0:
        print(out)
    err = stderr.decode("utf-8")
    if len(err) > 0:
        logging.error(err)

    tmpfile.close()
    tmp_qrels_file.close()
