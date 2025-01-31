import argparse


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="A simple knowledge base system")

    # Define sub-commands
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # Consume document sub-command
    subparsers.add_parser("consume", help="Consume a document")

    # Give answer sub-command
    give_answer_parser = subparsers.add_parser(
        "answer", help="Give an answer to a user's question"
    )
    give_answer_parser.add_argument(
        "--question", type=str, required=True, help="User's question"
    )

    # Generate question sub-command
    generate_question_parser = subparsers.add_parser(
        "generate", help="Generate a question for a chapter or topic"
    )
    generate_question_parser.add_argument(
        "--chapter", type=str, required=True, help="Chapter name"
    )
    generate_question_parser.add_argument(
        "--topic", type=str, required=False, help="Topic name (optional)"
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.subcommand == "consume":
        print(args.document)
    elif args.subcommand == "answer":
        print(args.question, args.document)
    elif args.subcommand == "generate":
        print(args.chapter, args.topic)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
