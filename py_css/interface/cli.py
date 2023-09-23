from rich.prompt import Prompt
from rich.style import Style
from rich.console import Console

import pyterrier as pt

import indexer.index as index_module
import models.baseline as baseline_module

index = None

pipeline: pt.Transformer


def process_input(input_str: str) -> str:
    """
    Process the input string.
    For now, just return the same string.

    Parameters
    ----------
    input_str : str
        The unprocessed user input string (query).

    Returns
    -------
    str
        The output to be shown to the user.
    """
    global index
    global pipeline

    result = pipeline.search(input_str)

    if result.empty:
        return "No Results Found"

    # Get the docno of the top-ranked document
    top_docno = int(result.iloc[0]["docno"])

    # Get the document content
    doc_content = index_module.get_document_content(top_docno)

    if doc_content is None:
        return "Internal Error: Top Document was not Found"

    return doc_content


def main(*, recreate: bool) -> None:
    """
    The main function of the CLI interface.

    Parameters
    ----------
    recreate : bool
        Whether to recreate the index.
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
            output = process_input(user_input)
            console.print(output, style=Style(italic=True))
