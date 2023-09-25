# Python Conversational Search System

A conversational search system built in python.

## Installation
Pull the repository from github, and install as a python package:
```bash
pip install -e .
```

## Usage
### CLI
If installed as a python package, the following command is available:
```bash
py_css cli
```

Otherwise, the equivalent can be achieved by navigating into the repository and running the following:
```bash
python py_css/main.py cli
```

### Run Queries File
```bash
python py_css/main.py run_file --log=INFO --queries=data/queries_train.csv --output=output/train.txt
```
