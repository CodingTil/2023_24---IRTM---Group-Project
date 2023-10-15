import signal
import sys
from typing import List, Tuple, Dict
import re
from pathlib import Path

from tqdm import tqdm
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from more_itertools import chunked
import pandas as pd


MODEL_NAME: str = "castorini/doc2query-t5-large-msmarco"
MAX_LENGTH: int = 512
BATCH_SIZE: int = 64
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES: int = 3

INPUT_FILE: Path = Path("data/collection.tsv")
OUTPUT_FILE: Path = Path("data/doc2query.tsv")


class Doc2Query:
    """
    A class for generating queries from documents using T5.

    Attributes
    ----------
    model : T5ForConditionalGeneration
        The T5 model to use
    tokenizer : T5TokenizerFast
        The T5 tokenizer to use
    max_length : int
        The maximum length of the input
    batch_size : int
        The batch size to use
    input_file : Path
        The input file
    output_file : Path
        The output file
    output_df : pd.DataFrame
        The output dataframe
    """

    model: T5ForConditionalGeneration
    tokenizer: T5TokenizerFast
    device: torch.device
    max_length: int
    num_samples: int
    batch_size: int
    input_file: Path
    output_file: Path
    output_df: pd.DataFrame
    pattern: re.Pattern

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = MAX_LENGTH,
        batch_size: int = BATCH_SIZE,
        device: str = DEVICE,
        num_samples: int = NUM_SAMPLES,
        input_file: Path = INPUT_FILE,
        output_file: Path = OUTPUT_FILE,
    ):
        """
        Constructor for the Doc2Query class.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to use, by default MODEL_NAME
        max_length : int, optional
            The maximum length of the input, by default MAX_LENGTH
        batch_size : int, optional
            The batch size to use, by default BATCH_SIZE
        device : str, optional
            The device to use, by default DEVICE
        num_samples : int, optional
            The number of samples to generate, by default NUM_SAMPLES
        input_file : Path, optional
            The input file, by default INPUT_FILE
        output_file : Path, optional
            The output file, by default OUTPUT_FILE
        """
        signal.signal(
            signal.SIGINT, lambda signo, _: self.__del__() and sys.exit(signo)
        )
        signal.signal(
            signal.SIGTERM, lambda signo, _: self.__del__() and sys.exit(signo)
        )

        self.device = torch.device(device)
        self.model = (
            T5ForConditionalGeneration.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name,
            legacy=False,
            model_max_length=max_length,
        )
        self.max_length = max_length
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.output_file = output_file
        assert input_file.exists()
        self.input_file = input_file
        if self.output_file.exists():
            self.output_df = pd.read_table(
                self.output_file,
                names=["docid"] + [f"query_{i}" for i in range(num_samples)],
                header=None,
            )
        else:
            self.output_df = pd.DataFrame(
                columns=["docid"] + [f"query_{i}" for i in range(num_samples)]
            )
        self.pattern = re.compile("^\\s*http\\S+")

    def add_new_queries(self, new_queries: List[Tuple[str, List[str]]]):
        """
        Add new queries to the output dataframe.

        Parameters
        ----------
        new_queries : List[Tuple[str, List[str]]]
            The new queries to add: (docid, queries)
        """
        new_data: Dict[str, List[str]] = {
            "docid": [],
            "query_0": [],
            "query_1": [],
            "query_2": [],
        }

        for docid, queries in new_queries:
            assert 1 <= len(queries) <= self.num_samples

            if len(queries) == self.num_samples:
                # We do not append the queries if they are already in the output dataframe
                # remove docid from output_df
                if docid in self.output_df["docid"].values:
                    self.output_df = self.output_df[self.output_df["docid"] != docid]
                new_data["docid"].append(docid)
                for i, query in enumerate(queries):
                    new_data[f"query_{i}"].append(query)
            else:
                assert docid in self.output_df["docid"].values
                # We append the queries if they are not in the output dataframe
                existing_queries: List[str] = []
                for i in range(self.num_samples):
                    # fetch the existing queries, if they are not NaN / None / strip() == "" etc.
                    query = self.output_df[self.output_df["docid"] == docid][
                        f"query_{i}"
                    ].values[0]
                    if query is not None and query.strip() != "":
                        existing_queries.append(query)

                assert len(existing_queries) + len(queries) == self.num_samples

                # remove docid from output_df
                self.output_df = self.output_df[self.output_df["docid"] != docid]
                new_data["docid"].append(docid)
                for i, query in enumerate(existing_queries + queries):
                    new_data[f"query_{i}"].append(query)

        self.output_df = pd.concat(
            [self.output_df, pd.DataFrame(new_data)], ignore_index=True
        )

    def write_output(self):
        self.output_df.to_csv(self.output_file, sep="\t", index=False, header=False)

    def __del__(self):
        self.write_output()

    def generate_queries(self):
        """
        Generate queries for the input file.
        """
        input_df = pd.read_table(
            self.input_file, names=["docid", "document"], header=None
        )
        # remove docids that are already in the output dataframe, that do not have any NaN/None/strip() == "" values
        skipping_ids: int = 0
        valid_docids = set(self.output_df["docid"])
        for _, row in self.output_df.iterrows():
            for i in range(self.num_samples):
                query = row[f"query_{i}"]
                if query is None or query.strip() == "":
                    valid_docids.remove(row["docid"])
                    skipping_ids += 1
                    break
        input_df = input_df[~input_df["docid"].isin(valid_docids)]

        print(
            f"Processing {len(input_df)} documents (skipping {skipping_ids}). Minimum ID: {input_df['docid'].min()}, maximum ID: {input_df['docid'].max()}"
        )

        self._generate_queries(
            list(zip(input_df["docid"].values, input_df["document"].values))
        )

    def _generate_queries(self, documents: List[Tuple[str, str]]):
        """
        Generate queries for a list of documents.

        Parameters
        ----------
        documents : List[Tuple[str, str]]
            The list of documents: (docid, document)

        Returns
        -------
        List[Tuple[str, List[str]]]
            The list of queries: (docid, queries)
        """
        iterator = chunked(iter(documents), self.batch_size)
        for batch in tqdm(iterator, total=len(documents) // self.batch_size + 1):
            docids: List[str] = []
            docs: List[str] = []
            for docid, doc in batch:
                docids.append(docid)
                docs.append(doc)
            queries = self._doc2query(docs)
            new_queries: List[Tuple[str, List[str]]] = list(zip(docids, queries))
            self.add_new_queries(new_queries)
            self.write_output()

    def _doc2query(self, texts: List[str]) -> List[List[str]]:
        """
        Generate queries for a list of texts.

        Parameters
        ----------
        texts : List[str]
            The list of texts

        Returns
        -------
        List[List[str]]
            The list of num_samples queries
        """
        docs = [re.sub(self.pattern, "", doc) for doc in texts]

        with torch.no_grad():
            input_ids = self.tokenizer(
                docs,
                max_length=self.max_length,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).input_ids.to(self.device)
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                do_sample=True,
                top_k=10,
                num_return_sequences=self.num_samples,
            )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        rtr = [gens for gens in chunked(outputs, self.num_samples)]
        return rtr


if __name__ == "__main__":
    Doc2Query().generate_queries()
