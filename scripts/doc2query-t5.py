from typing import List, Tuple, Dict, Set
import re
from pathlib import Path
from multiprocessing import Process, Queue

from tqdm import tqdm
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from more_itertools import chunked
import pandas as pd


MODEL_NAME: str = "castorini/doc2query-t5-large-msmarco"
MAX_LENGTH: int = 256
BATCH_SIZE: int = 32
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES: int = 3

INPUT_FILE: Path = Path("data/collection.tsv")
OUTPUT_FILE: Path = Path("data/doc2query.tsv")


def batched_writer(queue: Queue, output_file: Path, num_samples: int):
    while True:
        items_to_write = []

        # Wait till told to capture new queries
        item = queue.get()
        if item == "STOP":
            break

        # Append the first item
        items_to_write.append(item)

        # Now, try to get other items without blocking if the queue is empty
        break_after = False
        while not queue.empty():
            item = queue.get_nowait()
            if item == "STOP":
                break_after = True
                break
            items_to_write.append(item)

        # Write all collected items to the output file
        new_data = {
            "docid": [docid for docid, _ in items_to_write],
        }

        for i in range(num_samples):
            new_data[f"query_{i}"] = [
                queries[i] if i < len(queries) else "" for _, queries in items_to_write
            ]

        df = pd.DataFrame(new_data)
        df.to_csv(output_file, mode="a", sep="\t", index=False, header=False)

        if break_after:
            break


class Doc2Query:
    """
    A class for generating queries from documents using T5.

    Attributes
    ----------
    model : T5ForConditionalGeneration
        The T5 model to use
    tokenizer : T5TokenizerFast
        The T5 tokenizer to use
    device : torch.device
        The device to use
    max_length : int
        The maximum length of the input
    num_samples : int
        The number of samples to generate
    batch_size : int
        The batch size to use
    input_file : Path
        The input file
    output_file : Path
        The output file
    output_df : pd.DataFrame
        The output dataframe
    pattern : re.Pattern
        The pattern to remove URLs from the input
    write_queue : Queue
        The queue to write to
    writer_process : Process
        The process to write to the output file
    """

    model: T5ForConditionalGeneration
    tokenizer: T5TokenizerFast
    device: torch.device
    max_length: int
    num_samples: int
    batch_size: int
    input_file: Path
    output_file: Path
    pattern: re.Pattern
    write_queue: Queue
    writer_process: Process

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
        self.pattern = re.compile("^\\s*http\\S+")
        self.write_queue = Queue()
        self.writer_process = Process(
            target=batched_writer,
            args=(self.write_queue, self.output_file, self.num_samples),
        )
        self.writer_process.start()

    def add_new_queries(self, new_queries: List[Tuple[str, List[str]]]):
        """
        Add new queries to the output dataframe.

        Parameters
        ----------
        new_queries : List[Tuple[str, List[str]]]
            The new queries to add: (docid, queries)
        """
        for docid, queries in new_queries:
            self.write_queue.put((docid, queries))

    def _already_processed_docids(self) -> Set[int]:
        """Get set of docids that have already been processed."""
        if not self.output_file.exists():
            return set()

        with open(self.output_file, "r") as f:
            # Reading only the docid column (1st column) and returning it as a set
            return set(int(line.split("\t")[0]) for line in f.readlines())

    def generate_queries(self):
        """
        Generate queries for the input file.
        """
        input_df = pd.read_table(
            self.input_file, names=["docid", "document"], header=None
        )

        processed_docids = self._already_processed_docids()

        skipping_ids = input_df["docid"].nunique()
        input_df = input_df[~input_df["docid"].isin(processed_docids)]
        skipping_ids -= input_df["docid"].nunique()

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
        self.write_queue.put("STOP")

    def __del__(self):
        self.write_queue.put("STOP")
        self.writer_process.join()

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

    def sort_output_file(self):
        """Sort the output file by docid."""
        if not self.output_file.exists():
            print("Output file does not exist.")
            return

        df = pd.read_table(
            self.output_file,
            names=["docid"] + [f"query_{i}" for i in range(self.num_samples)],
            header=None,
        )
        df = df.sort_values(by="docid")
        df.to_csv(self.output_file, sep="\t", index=False, header=False)

    def verify_output(self):
        """Check if all docids from the input_file have queries in the output_file."""
        # Check if output file exists
        if not self.output_file.exists():
            print("Output file does not exist. Verification failed!")
            return

        input_df = pd.read_table(
            self.input_file, names=["docid", "document"], header=None
        )
        output_df = pd.read_table(
            self.output_file,
            names=["docid"] + [f"query_{i}" for i in range(self.num_samples)],
            header=None,
        )

        input_docids = set(input_df["docid"].values)
        output_docids = set(output_df["docid"].values)

        missing_docids = input_docids - output_docids

        if not missing_docids:
            print(
                "All docids from input_file have corresponding queries in the output_file."
            )
        else:
            print(
                f"Missing queries for {len(missing_docids)} docids in the output_file."
            )
            print("Some of the missing docids are:", list(missing_docids)[:10])


if __name__ == "__main__":
    d2q = Doc2Query()
    d2q.generate_queries()
    d2q.sort_output_file()
    d2q.verify_output()
