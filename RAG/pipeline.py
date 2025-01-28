import ollama
import asyncio
from utils import RAGtoolkit
from pydantic import BaseModel


client = ollama.AsyncClient()


class SelectedDocument(BaseModel):
    file_name: str
    chapter_num: int
    target_questions: list[str]


class SelectedDocumentList(BaseModel):
    documents: list[SelectedDocument]


class QuestionAnswer(BaseModel):
    question: str
    answer: str


document_selection_instruction_template = """
You're supposed to read the following context, which will be summaries of multiple documents,

<document-metadatas>
{}
</document-metadatas>

then look at the question and select documents from the list,

<question>
{}
</question>

reference, what part of the question, the selected document helps to answer,
and give me the result.

NOTE: decompose the initial question into sub-questions if needed
"""


question_answering_for_chunks_template = """
I will provide you chunks of documents, which will serve as context,
that you shall use, to answer my question

<document-chunks>
{}
</document-chunks>

use these chunks to answer the following question

<question>
{}
</question>
"""


question_answering_merge_template = """
I will provide you sub-questions,
for which the following answers have been derived

<questions-answers>
{}
</questions-answers>

using the following context, answer this question

<question>
{}
</question>
"""


async def DocumentQnA(question: str) -> QuestionAnswer:
    # step 1 get all metadata, use it in tandem with question, to decide which documents to use
    generic_kit = RAGtoolkit()
    metadatas = generic_kit.get_all_metadata()
    res = await client.chat(
        model="deepseek-r1",
        messages=[
            {
                "role": "user",
                "content": document_selection_instruction_template.format(
                    metadatas, question
                ),
            },
        ],
        format=SelectedDocumentList.model_json_schema(),
        options={"num_ctx": 8192, "temperature": 0.5},
        keep_alive=0,
    )

    docs = SelectedDocumentList.model_validate_json(res.message.content).documents

    print(docs)

    qna_list: list[QuestionAnswer] = []

    # step 2 using these documents, answer the sub-questions, that the initial question was broken into
    for doc in docs:
        questions = doc.target_questions
        chapter = doc.chapter_num

        chapter_toolkit = RAGtoolkit(chapter_num=chapter)

        for question in questions:
            chapter_chunks = chapter_toolkit.query_docs(question, 10)["documents"][0]

            res = await client.chat(
                model="deepseek-r1",
                messages=[
                    {
                        "role": "user",
                        "content": question_answering_for_chunks_template.format(
                            "\n--chunk--\n".join(chapter_chunks), question
                        ),
                    }
                ],
                format=QuestionAnswer.model_json_schema(),
                options={"num_ctx": 16384},
                keep_alive=0,
            )

            question_answer = QuestionAnswer.model_validate_json(res.message.content)
            qna_list.append(question_answer)

    # step 3 merge result from these answers, and answer the initial question
    res = await client.chat(
        model="deepseek-r1",
        messages=[
            {
                "role": "user",
                "content": question_answering_merge_template.format(
                    "\n---chunk---\n".join(
                        [
                            f"<question>{qna.question}</question>\n<answer>{qna.answer}</answer>"
                            for qna in qna_list
                        ]
                    ),
                    question,
                ),
            }
        ],
        format=QuestionAnswer.model_json_schema(),
        options={"num_ctx": 16384},
        keep_alive=0,
    )
    return QuestionAnswer.model_validate_json(res.message.content)


async def main():
    ans = await DocumentQnA("what is a transition function?")
    print(ans)
    """
    output:
    question='What is a transition function?'
    answer="The transition function describes how an automaton changes its state based on input.
    It's denoted by δ(q, a) = p, where q and p are states and a is an input symbol."
    """


if __name__ == "__main__":
    asyncio.run(main())


"""
2. Questions generation pipeline -> used to generate questions on per chapter basis, or per topic basis
3. Answers generation pipeline -> used to generate answers for each of the question
"""
