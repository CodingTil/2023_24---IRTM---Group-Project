import os
from pkg_resources import resource_filename
import shutil
import logging
from typing import Dict, Generator

import pyterrier as pt

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

    # Create an index with both "docno" and "text" as metadata
    logging.info("Creating Index")
    iter_indexer = pt.IterDictIndexer(INDEX_PATH, verbose=True, blocks=True)
    return iter_indexer.index(
        document_collection_generator(), meta={"docno": 20, "text": 4096}
    )


def document_collection_generator() -> Generator[Dict[str, str], None, None]:
    """
    Return a generator over the document collection.

    Yields
    -------
    Generator[Dict[str, str], None, None]
        A generator over the document collection (docno, content).
    """
    with open(DATA_PATH, "r") as f:
        for line in f:
            docno, content = line.split("\t")
            if content.strip() == "":
                continue
            yield {"docno": docno, "text": content}
