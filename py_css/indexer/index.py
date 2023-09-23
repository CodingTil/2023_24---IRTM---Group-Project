import os
from pkg_resources import resource_filename
import shutil
import pyterrier as pt
import pandas as pd

# Paths for data and index
DATA_PATH = resource_filename(__name__, "../../data/collection.tsv")
INDEX_PATH = resource_filename(__name__, "../../data/index")

def get_index(*, recreate: bool=False):
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
        shutil.rmtree(INDEX_PATH)

    # Load the dataset
    dataset = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["docno", "text"])
    dataset['docno'] = dataset['docno'].astype(str)

    # Create an index
    index = pt.DFIndexer(INDEX_PATH)
    return index.index(dataset.text, dataset.docno)

