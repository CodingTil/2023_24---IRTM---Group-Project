import logging
import string

from rich.prompt import Prompt
from rich.style import Style
from rich.console import Console

import pyterrier as pt

import indexer.index as index_module
import models.baseline as baseline_module

index = None

pipeline: pt.Transformer


def process_input(input_str: str, *, top_n: int) -> str:
    """
    Process the input string.
    For now, just return the same string.

    Parameters
    ----------
    input_str : str
        The unprocessed user input string (query).
    top_n : int
        The number of top-ranked documents to return.

    Returns
    -------
    str
        The output to be shown to the user.
    """
    global index
    global pipeline

    # remove punctuation
    input_str = input_str.translate(str.maketrans("", "", string.punctuation))

    result = pipeline.search(input_str)

    if result.empty:
        return "No Results Found"

    # Get the docno of the top_n top-ranked documents
    top_docs = result["docno"].head(top_n).tolist()
    top_docs = [int(docno) for docno in top_docs]

    logging.info(f"Top {top_n} Documents: {top_docs}")

    # Get the document content
    contents = index_module.get_documents_content(top_docs)

    contents = [c for c in contents if c is not None]

    if len(contents) == 0:
        return "Internal Error: Top Document was not found"

    return "\n".join(contents)


def main(*, recreate: bool, top_n: int) -> None:
    """
    The main function of the CLI interface.

    Parameters
    ----------
    recreate : bool
        Whether to recreate the index.
    top_n : int
        The number of top-ranked documents to return.
    """
    global index
    global pipeline

    index = index_module.get_index(recreate=recreate)
    pipeline = baseline_module.create_pipeline(index)

    # Initialize the rich console
    console = Console()

    # Display instructions using rich's print with color
    console.print(
        "Welcome to the Conversational Search Engine (CSE)!", style="bold blue"
    )
    console.print(
        "Enter your queries below. To exit, type [bold red]exit[/bold red] and press Enter.\n",
        style="blue",
    )

    while True:
        # Using rich's prompt to get input
        user_input = Prompt.ask("Query", console=console)

        # Check for exit condition
        if user_input.lower() == "exit":
            break

        # Check if user input is non-empty
        if user_input.strip():
            output = process_input(user_input, top_n=top_n)
            console.print(output, style=Style(italic=True))
