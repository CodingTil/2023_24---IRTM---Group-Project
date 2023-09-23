import argparse
import logging

import pyterrier as pt

import interface.cli as cli_module


def setup() -> None:
    """
    Set up the necessary configurations.
    """
    if not pt.started():
        pt.init()


def main():
    parser = argparse.ArgumentParser(
        description="Entry point for the Conversational Search Engine (CSE)"
    )

    # --log=XYZ for log level
    parser.add_argument(
        "--log",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set the logging level",
    )

    # Define the command argument
    parser.add_argument(
        "command",
        type=str,
        choices=["cli"],
        help='Command to run (e.g., "cli" for command line interface)',
    )

    # --recreate for recreating the index
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the index",
    )

    # --top_n=123 for the number of top-ranked documents to return
    parser.add_argument(
        "--top_n",
        type=int,
        default=3,
        help="The number of top-ranked documents to return",
    )

    args = parser.parse_args()

    # Log Level
    logging.basicConfig(level=args.log)

    # Call the setup function
    setup()

    # Check the provided command and act accordingly
    if args.command == "cli":
        cli_module.main(recreate=args.recreate, top_n=args.top_n)


if __name__ == "__main__":
    main()
