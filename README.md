# Python Conversational Search System

A conversational search system built in python.

## Installation
Pull the repository from github, and install as a python package:
```bash
pip install -e .
```

## Usage
If installed locally, henceforth the command `py_css` is available. Otherwise, the following entrypoint shall be called:
```bash
python py_css/main.py
# OR, if installed locally:
py_css
```

A detailed help page will be presented using:
```bash
py_css --help
```

### CLI Mode
If installed as a python package, the following command is available:
```bash
py_css cli
```

### Run Queries File
```bash
py_css run_file --log=INFO --queries=data/queries_train.csv --output=output/train.txt
```

### Run Queries and Evaluate Performance
```bash
py_css eval --log=INFO --queries=data/queries_train.csv --qrels=data/qrels_train.txt
```

### Create Kaggle Runfile Format
```bash
py_css kaggle --log=INFO --queries=data/queries_test.csv --output=output/kaggle-prf.csv
```


## Retrieval Pipelines
As outlined in the paper, four retrieval pipelines were implemented:

### Baseline
Can be selected by specifying the following parameters:
```bash
--method=baseline
--baseline-params=1000,1000,50
```

#### Indexing
For indexing, the document collection has to be placed into the `data/` folder.
<br>
[Further Instructions](data/README.md)

#### Parameters
| Position | ID | Description | Constraints |
| --- | --- | --- | --- |
| 0 | `bm25_docs` | The number of documents to be retrieved using `BM25`. | |
| 1 | `mono_t5_docs` | The number of documents to be reranked by `monoT5` after retrieval. | `bm25_docs >= mono_t5_docs` |
| 2 | `duo_t5_docs` | The number of documents to be reranked by `duoT5` after `monoT5` reranking. | `mono_t5_docs <= duo_t5_docs` |

### Baseline + `RM3`

Can be selected by specifying the following parameters:
```bash
--method=baseline-prf
--baseline-prf-params=1000,17,26,1000,50
```

#### Indexing
For indexing, the document collection has to be placed into the `data/` folder.
<br>
[Further Instructions](data/README.md)

#### Parameters
| Position | ID | Description | Constraints |
| --- | --- | --- | --- |
| 0 | `bm25_docs` | The number of documents to be retrieved using `BM25`. | |
| 1 | `rm3_fb_docs` | The number of documents to be used for `RM3` query expansion. | |
| 2 | `rm3_fb_terms` | The number of terms to expand the query with using `RM3`. | |
| 3 | `mono_t5_docs` | The number of documents to be reranked by `monoT5` after retrieval. | `bm25_docs >= mono_t5_docs` |
| 4 | `duo_t5_docs` | The number of documents to be reranked by `duoT5` after `monoT5` reranking. | `mono_t5_docs <= duo_t5_docs` |


### `doc2query`
Can be selected by specifying the following parameters:
```bash
--method=doc2query
--doc2query-params=1000,1000,50
```

#### Indexing
For indexing, the document collection has to be placed into the `data/` folder.
Additionally, descriptive queries for each document have to be generated using [this script](scripts/doc2query-t5.py).
<br>
[Further Instructions](data/README.md)

#### Parameters
| Position | ID | Description | Constraints |
| --- | --- | --- | --- |
| 0 | `bm25_docs` | The number of documents to be retrieved using `BM25`. | |
| 1 | `mono_t5_docs` | The number of documents to be reranked by `monoT5` after retrieval. | `bm25_docs >= mono_t5_docs` |
| 2 | `duo_t5_docs` | The number of documents to be reranked by `duoT5` after `monoT5` reranking. | `mono_t5_docs <= duo_t5_docs` |

### `doc2query` + `RM3`

Can be selected by specifying the following parameters:
```bash
--method=doc2query-prf
--doc2query-prf-params=1000,17,26,1000,50
```

#### Indexing
For indexing, the document collection has to be placed into the `data/` folder.
Additionally, descriptive queries for each document have to be generated using [this script](scripts/doc2query-t5.py).
<br>
[Further Instructions](data/README.md)

#### Parameters
| Position | ID | Description | Constraints |
| --- | --- | --- | --- |
| 0 | `bm25_docs` | The number of documents to be retrieved using `BM25`. | |
| 1 | `rm3_fb_docs` | The number of documents to be used for `RM3` query expansion. | |
| 2 | `rm3_fb_terms` | The number of terms to expand the query with using `RM3`. | |
| 3 | `mono_t5_docs` | The number of documents to be reranked by `monoT5` after retrieval. | `bm25_docs >= mono_t5_docs` |
| 4 | `duo_t5_docs` | The number of documents to be reranked by `duoT5` after `monoT5` reranking. | `mono_t5_docs <= duo_t5_docs` |
