import ollama
import asyncio
from .utils import RAGtoolkit, grade_statement_similarity
import os
import json

from .llm_utils import (
    decompose_question,
    QuestionAnswer,
    QuestionsFromTag,
    answer_atomic_question,
    answer_composite_question,
    generate_questions,
)


ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

client = ollama.AsyncClient(host=ollama_url)


"""
Pipelines
1. Question Generation from Topics ✅
2. Answer Question Pipeline ✅
3. Evaluate user answers Pipeline ✅
"""


async def question_answer_pipeline(question: str, topic: str) -> QuestionAnswer:
    """
    Generates a composite answer for a question and saves the QnA pair in a JSON file
    at generated/{topic}/QnA.json. Before running the heavy asynchronous processing,
    it checks if a similar question already exists (using grade_statement_similarity > 90%).

    Args:
        question (str): The input question.
        topic (str): The topic name used for file storage.

    Returns:
        QuestionAnswer: The QnA pair, either retrieved from storage or newly generated.
    """
    # Define the output file path for QnA.
    output_folder = os.path.join("generated", topic)
    os.makedirs(output_folder, exist_ok=True)
    output_filepath = os.path.join(output_folder, "QnA.json")

    # Load existing QnA entries if the file exists.
    existing_entries = []
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding="utf-8") as f:
            existing_entries = json.load(f)
        # Check for a duplicate by comparing the input question with each stored question.
        for entry in existing_entries:
            similarity = grade_statement_similarity(question, entry["question"])
            if similarity > 95:
                # If a duplicate is found, return the existing QnA pair.
                return QuestionAnswer(**entry)

    # No duplicate found: proceed with generating the answer.
    decomposed_question = await decompose_question(question=question)

    chain_of_questions: list[str] = []
    atomic_answers: list[QuestionAnswer] = []

    for q in decomposed_question.sub_questions:

        atomic_answer = await answer_atomic_question(q, topic, chain_of_questions)

        chain_addition = f"""
        Question:
            {atomic_answer.question}
        Answer:
            {atomic_answer.answer}
        """

        print(chain_addition)

        chain_of_questions.append(chain_addition)
        atomic_answers.append(atomic_answer)

    # Generate the composite answer.
    answer = await answer_composite_question(question, atomic_answers)

    # Convert the new answer (a QuestionAnswer Pydantic model) to a dict.
    new_entry = answer.model_dump()

    # Append the new QnA entry to the list.
    existing_entries.append(new_entry)

    # Save the updated QnA list back to the file.
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(existing_entries, f, indent=4, ensure_ascii=False)

    return answer


async def questions_generation_pipeline(
    topic: str, tags: list[str], chapter_num: int = 0
) -> list[QuestionsFromTag]:
    # Determine output filename based on tags presence
    if len(tags) == 0:
        kit = RAGtoolkit(topic=topic)
        tags = kit.get_chapter_tags(chapter_num)
        print(tags)
        filename = f"chapter_{chapter_num}.json"
    else:
        filename = "user-defined.json"

    payloads = [(tag, topic) for tag in tags]

    # Generate questions concurrently for each tag
    questions = []

    for payload in payloads:
        question = await generate_questions(*payload)
        print(question)
        questions.append(question)

    # Create the output directory if it doesn't exist
    output_folder = os.path.join("generated", topic, "questions")
    os.makedirs(output_folder, exist_ok=True)
    output_filepath = os.path.join(output_folder, filename)

    # Convert each QuestionsFromTag (a pydantic BaseModel) to a dict
    new_data = [q.model_dump() for q in questions]

    # Load existing data if file exists; otherwise, start with an empty list
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    # Build a mapping from tag to its existing data for easy lookup
    existing_map = {item["tag"]: item for item in existing_data}

    # Merge new questions into existing data
    for new_item in new_data:
        tag = new_item["tag"]
        new_questions = new_item["questions"]
        if tag in existing_map:
            # For each new question, check if a similar one (>=90% similarity) exists
            for new_q in new_questions:
                duplicate_found = False
                for existing_q in existing_map[tag]["questions"]:
                    if grade_statement_similarity(new_q, existing_q) >= 90:
                        duplicate_found = True
                        break
                if not duplicate_found:
                    existing_map[tag]["questions"].append(new_q)
        else:
            # No existing data for this tag: add it directly
            existing_map[tag] = new_item

    # Convert the merged mapping back to a list
    merged_data = list(existing_map.values())

    # Write the merged data back to the file (using ensure_ascii=False to preserve special characters)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    return questions


async def evaluate_question_answer(topic: str, question: str, user_answer: str) -> dict:
    """
    Evaluates a user's answer by generating a reference answer using the question_answer_pipeline
    and then comparing it with the user's answer using grade_statement_similarity.

    Args:
        topic (str): The topic to which the question belongs.
        question (str): The question to be answered.
        user_answer (str): The answer provided by the user.

    Returns:
        dict: A dictionary containing the similarity score (percentage) and the generated reference answer.
    """
    # Generate the reference answer using your existing composite question-answer pipeline.
    reference_qa = await question_answer_pipeline(question, topic)

    # Use grade_statement_similarity to compute the similarity score between the user's answer
    # and the generated reference answer. This score is on a 0-100% scale.
    similarity_score = grade_statement_similarity(user_answer, reference_qa.answer)

    return {"score": similarity_score, "reference_answer": reference_qa.answer}


async def main():
    answer = await question_answer_pipeline(
        "What does Δ̂(P, s_1 s_2 s_3 ⋯ s_n) represent in the terms of NFA processing?",
        "automata",
    )
    print(answer)
    # await questions_generation_pipeline("automata", [], 1)


# asyncio.run(questions_generation_pipeline("automata", tags=[], chapter_num=0))
