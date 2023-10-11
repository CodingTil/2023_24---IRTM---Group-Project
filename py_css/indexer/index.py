import os
from pkg_resources import resource_filename
import shutil
import logging
from typing import Dict, Generator

import pandas as pd
import pyterrier as pt
from pyterrier_doc2query import Doc2Query

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
    df = document_collection_dataframe()
    df = add_query_description_to_documents(df)
    df_indexer = pt.DFIndexer(INDEX_PATH, verbose=True, blocks=True)
    logging.info("Indexing")
    return df_indexer.index(df["text"], df)


def document_collection_dataframe() -> pd.DataFrame:
    """
    Return a dataframe of the document collection.

    Returns
    -------
    pd.DataFrame
        A dataframe of the document collection.
    """
    df = pd.read_table(
        DATA_PATH, names=["docno", "text"], header=0, dtype={"docno": str, "text": str}
    )
    # Remove the rows where text is empty
    df = df[df["text"].str.strip().astype(bool)]
    return df


def add_query_description_to_documents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a query description to each document.

    Parameters
    ----------
    pd.DataFrame
        A dataframe of the document collection.

    Returns
    -------
    pd.DataFrame
        A dataframe of the document collection with a query description.
    """
    doc_2_query_t5 = Doc2Query(
        append=False,
        out_attr="querygen",
        fast_tokenizer=True,
        verbose=True,
        num_samples=1,
    )
    logging.info("Adding query description to documents")
    return doc_2_query_t5.transform(df)
