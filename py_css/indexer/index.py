import os
from pkg_resources import resource_filename
import shutil
import pyterrier as pt
import pandas as pd
import logging
from typing import Optional

# Paths for data and index
DATA_PATH = resource_filename(__name__, "../../data/collection.tsv")
INDEX_PATH = resource_filename(__name__, "../../data/index")


def get_index(*, recreate: bool = False):
    """
    Return a PyTerrier Index.

    If the index is not found, create it using the document collection.

    Parameters
    ----------
    recreate : bool, optional
        Whether to recreate the index even if it exists, by default False

    Returns
    -------
    pt.Index
        The PyTerrier Index.
    """
    index_ref: pt.IndexRef
    # Check if the index exists
    if not os.path.exists(INDEX_PATH) or recreate:
        # Index does not exist, so create it
        index_ref = create_index()
    else:
        # Index exists, so load it
        logging.info("Loading Index")
        index_ref = pt.IndexRef.of(os.path.join(INDEX_PATH, "data.properties"))
    return pt.IndexFactory.of(index_ref)


def create_index() -> pt.IndexRef:
    """
    Create a PyTerrier index using the document collection.

    Returns
    -------
    pt.IndexRef
        The PyTerrier Index.
    """
    # Initialize PyTerrier if not done yet
    if not pt.started():
        pt.init()

    # if the index exists, delete it
    if os.path.exists(INDEX_PATH):
        logging.info("Recreating Index")
        shutil.rmtree(INDEX_PATH)

    # Load the dataset
    logging.info("Reading Document Collection")
    dataset = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["docno", "text"])
    dataset["docno"] = dataset["docno"].astype(str)

    # Create an index with both "docno" and "text" as metadata
    logging.info("Creating Index")
    indexer = pt.DFIndexer(INDEX_PATH, blocks=True, verbose=True)
    return indexer.index(dataset.text, dataset.docno)


def get_document_content(docno: int) -> Optional[str]:
    """
    Get the document content of a specific document.

    Parameters
    ----------
    docno : int
        The document number.

    Returns
    -------
    Optional[str]
        The document content, or None if the document is not found.
    """
    # without using index
    with open(DATA_PATH, "r") as f:
        for line in f:
            if line.startswith(str(docno)):
                return line.split("\t")[1]
    return None
