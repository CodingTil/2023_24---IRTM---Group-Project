import argparse
import logging

import pyterrier as pt

import interface.cli as cli_module
import interface.run_queries as run_queries_module


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
        choices=["cli", "run_file"],
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

    # Command run_file with argument --queries=filepath --output=filepath
    parser.add_argument(
        "--queries",
        type=str,
        help="The path to the queries file",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="The path to the output file",
    )

    args = parser.parse_args()

    # Log Level
    logging.basicConfig(level=args.log)

    # Call the setup function
    setup()

    # Check the provided command and act accordingly
    if args.command == "cli":
        cli_module.main(recreate=args.recreate, top_n=args.top_n)
    elif args.command == "run_file":
        run_queries_module.main(
            recreate=args.recreate,
            queries_file_path=args.queries,
            output_file_path=args.output,
        )


if __name__ == "__main__":
    main()
