import argparse
import asyncio
import time

from RAG.extraction import dump_extraction_to_db
from RAG.utils import RAGtoolkit

from RAG.pipeline import (
    questions_generation_pipeline,
    evaluate_question_answer,
    question_answer_pipeline,
)


async def read_pipeline(
    topic: str, file_path: str, chapter_num: int, doc_structure: str
):
    from extraction.pipeline import async_extraction_pipeline_from_pdf

    print(
        f"Reading from file '{file_path}' for topic '{topic}', chapter {chapter_num}..."
    )
    start_time = time.time()
    await async_extraction_pipeline_from_pdf(
        file_path,
        subject=topic,
        chapter_num=chapter_num,
        document_structure=doc_structure,
    )
    end_time = time.time()
    print(f"time taken for document extraction --- {end_time - start_time} s---")
    dump_extraction_to_db(topic=topic)
    end_time = time.time()
    print(f"time taken for total extraction --- {end_time - start_time} s---")


async def summarize(topic: str, chapter_num: int):
    print(f"Summarizing topic '{topic}', chapter {chapter_num}...")
    kit = RAGtoolkit(topic=topic)

    summary = kit.get_chapter_summary(chapter_num=chapter_num)
    print(summary)


async def generate_questions_pipeline(
    topic: str, chapter_num: int, sub_topics: list[str]
):
    # Call your asynchronous questions_generation_pipeline
    if len(sub_topics) == 0:
        print(f"Generating questions for topic '{topic}' chapter {chapter_num}")
        questions = await questions_generation_pipeline(topic, [], chapter_num)

    else:
        print(
            f"Generating questions for topic '{topic}', with sub-topics: {sub_topics}"
        )
        questions = await questions_generation_pipeline(topic, tags=sub_topics)

    print("Generated questions:")
    for q in questions:
        print(q)


async def evaluation_pipeline(topic: str, question: str, user_answer: str):
    # Call your evaluate_question_answer pipeline and print the results.
    result = await evaluate_question_answer(topic, question, user_answer)
    print("Evaluation Results:")
    print(f"Score: {result['score']:.2f}%")
    print("Reference Answer:")
    print(result["reference_answer"])


async def answer_user_question(topic: str, question: str):
    # Call your evaluate_question_answer pipeline and print the results.
    result = await question_answer_pipeline(question, topic)
    print("Answer:")
    print(result.answer)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Toolkit Main Pipeline",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub-command to run"
    )

    # "read" subcommand
    read_parser = subparsers.add_parser("read", help="Read data from a file")
    read_parser.add_argument("--for", dest="topic", required=True, help="Topic name")
    read_parser.add_argument(
        "--from", dest="file_path", required=True, help="Path to input file"
    )
    read_parser.add_argument(
        "--chapter_num", type=int, required=True, help="Chapter number"
    )
    generate_parser.add_argument(
        "--structure",
        dest="doc_structure",
        required=False,
        help="Document Structure `research/default`",
    )

    # "generate" subcommand
    generate_parser = subparsers.add_parser(
        "generate", help="Generate questions for tags"
    )
    generate_parser.add_argument(
        "--for", dest="topic", required=True, help="Topic name"
    )
    generate_parser.add_argument(
        "--chapter_num", type=int, required=True, help="Chapter number"
    )
    generate_parser.add_argument(
        "--sub-topics",
        nargs="+",
        dest="sub_topics",
        required=True,
        help="Tags associated with the topic",
    )

    # "summarize" subcommand
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize chapter content"
    )
    summarize_parser.add_argument(
        "--for", dest="topic", required=True, help="Topic name"
    )
    summarize_parser.add_argument(
        "--chapter_num", type=int, required=True, help="Chapter number"
    )

    # "evaluate" subcommand
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a question-answer pair"
    )
    evaluate_parser.add_argument(
        "--for", dest="topic", required=True, help="Topic name"
    )
    evaluate_parser.add_argument("--question", required=True, help="Question string")
    evaluate_parser.add_argument(
        "--answer", required=True, help="User's answer to the question"
    )

    answer_parser = subparsers.add_parser("answer", help="Answer user's question")
    answer_parser.add_argument("--for", dest="topic", required=True, help="Topic name")
    answer_parser.add_argument(
        "--question", type=str, required=True, help="Question to answer, for user"
    )

    args = parser.parse_args()

    # Dispatch based on the subcommand provided.
    if args.command == "read":
        asyncio.run(
            read_pipeline(
                args.topic, args.file_path, args.chapter_num, args.doc_structure
            )
        )
    elif args.command == "generate":
        asyncio.run(
            generate_questions_pipeline(args.topic, args.chapter_num, args.sub_topics)
        )
    elif args.command == "summarize":
        asyncio.run(summarize(args.topic, args.chapter_num))
    elif args.command == "evaluate":
        asyncio.run(evaluation_pipeline(args.topic, args.question, args.answer))
    elif args.command == "answer":
        asyncio.run(answer_user_question(args.topic, args.question))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
