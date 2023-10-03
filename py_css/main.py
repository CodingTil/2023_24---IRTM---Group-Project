import argparse
import logging

import pyterrier as pt

import interface.cli as cli_module
import interface.run_queries as run_queries_module
import interface.eval as eval_module

import models.base as base_module
import models.baseline as baseline_module


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

    # Global arguments
    global_args = parser.add_argument_group("Global arguments")

    global_args.add_argument(
        "--log",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set the logging level",
    )

    global_args.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the index",
    )

    global_args.add_argument(
        "--method",
        type=str,
        choices=["baseline", "advanced"],
        default="baseline",
        help="Set the retrieval method",
    )

    global_args.add_argument(
        "--baseline-params",
        type=lambda s: tuple(map(int, s.split(","))),
        default=(1000, 100, 10),
        help="Parameters for baseline method as tuple (bm25_docs, mono_t5_docs, duo_t5_docs)",
    )

    # Command argument
    parser.add_argument(
        "command",
        type=str,
        choices=["cli", "run_file", "eval"],
        help='Command to run (e.g., "cli" for command line interface)',
    )

    # CLI mode arguments
    cli_args = parser.add_argument_group("CLI arguments")

    cli_args.add_argument(
        "--top_n",
        type=int,
        default=3,
        help="The number of top-ranked documents to return for CLI mode",
    )

    # Run file arguments
    run_file_args = parser.add_argument_group("Run File arguments")

    run_file_args.add_argument(
        "--queries",
        type=str,
        help="The path to the queries file",
    )

    run_file_args.add_argument(
        "--output",
        type=str,
        help="The path to the output file",
    )

    # Eval arguments
    eval_args = parser.add_argument_group("Eval arguments")

    eval_args.add_argument(
        "--qrels",
        type=str,
        help="The path to the qrels file",
    )

    args = parser.parse_args()

    # Log Level
    logging.basicConfig(level=args.log)

    # Call the setup function
    setup()

    # Check the provided command and act accordingly
    if args.command == "cli":
        cli_module.main(
            recreate=args.recreate,
            top_n=args.top_n,
            baseline_params=args.baseline_params,
        )
    elif args.command == "run_file":
        run_queries_module.main(
            recreate=args.recreate,
            queries_file_path=args.queries,
            output_file_path=args.output,
            baseline_params=args.baseline_params,
        )
    elif args.command == "eval":
        eval_module.main(
            recreate=args.recreate,
            queries_file_path=args.queries,
            qrels_file_path=args.qrels,
            baseline_params=args.baseline_params,
        )


if __name__ == "__main__":
    main()
