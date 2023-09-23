import argparse
import pyterrier as pt

import interface.cli as cli_module

def setup() -> None:
    """
    Set up the necessary configurations.
    """
    if not pt.started():
        pt.init()


def main():
    parser = argparse.ArgumentParser(description="Entry point for the Conversational Search Engine (CSE)")
    
    # Define the command argument
    parser.add_argument('command', type=str, choices=['cli'], help='Command to run (e.g., "cli" for command line interface)')

    args = parser.parse_args()

    # Call the setup function
    setup()

    # Check the provided command and act accordingly
    if args.command == "cli":
        cli_module.main()


if __name__ == "__main__":
    main()

