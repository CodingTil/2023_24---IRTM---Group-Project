import logging

from rich.prompt import Prompt
from rich.style import Style
from rich.console import Console

import pyterrier as pt

import indexer.index as index_module
import models.base as base_module
import models.baseline as baseline_module

index = None
pipeline: base_module.Pipeline

context: base_module.Context = []


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
    global context

    new_query_id = len(context) + 1
    new_query = base_module.Query(f"q_{new_query_id}", input_str)

    # Search for the query
    context, _ = pipeline.search(new_query, context)

    # Get the top_n top-ranked documents
    result = context[-1][1]

    if result is None or len(result) == 0:
        return "No Results Found"

    result = result[:top_n]

    # Get the docno of the top_n top-ranked documents
    top_docs = [int(r.docno) for r in result]

    logging.info(f"Top {top_n} Documents: {top_docs}")

    contents = [r.content for r in result]

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
    pipeline = baseline_module.Baseline(index)

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
