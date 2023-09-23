from rich.prompt import Prompt
from rich.style import Style
from rich.console import Console

import indexer.index as index_module

index = None

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
    # append some random statistic of the index
    return input_str + "\n" + index.getCollectionStatistics().toString()


def main() -> None:
    """
    The main function of the CLI interface.
    """
    global index
    index = index_module.get_index(recreate=False)

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

