import ollama
import asyncio
from utils import RAGtoolkit, convert_string_to_list
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
                options={"num_ctx": 16384, "temperature": 0.2},
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
        options={"num_ctx": 16384, "temperature": 0},
        keep_alive=0,
    )
    return QuestionAnswer.model_validate_json(res.message.content)


class Topics(BaseModel):
    file_name: str
    chapter_num: int
    topics: list[str]


class TopicList(BaseModel):
    topic_list: list[Topics]


class QuestionForTopic(BaseModel):
    topic: str
    questions: list[str]


topics_filteration_prompt = """
<context>
{}
</context>

You are supposed to use the given context,
to filter the topics I give to you, based on the context
into chapters that they're associated with

<topics>
{}
</topics>

the following are the topics you need sort,
into their required chapters

NOTE: be extremely strict with your sorting
"""

question_generation_prompt = """
<context>
{}
</context>

using the following context

<topic>
{}
</topic>

for the above topic

generate long answer questions, that can be asked for the topic

NOTE: limit the number of questions to "3"
"""


async def get_questions(chunks: list[str], topic: str) -> QuestionForTopic:
    res = await client.chat(
        model="deepseek-r1",
        messages=[
            {
                "role": "user",
                "content": question_generation_prompt.format(chunks, topic),
            }
        ],
        format=QuestionForTopic.model_json_schema(),
        options={"temperature": 0.3, "num_ctx": 8192},
        keep_alive=0,
    )
    output = QuestionForTopic.model_validate_json(res.message.content)
    return output


async def QuestionsGeneration(
    topics: list[str] = [], chapter_num: int = 0
) -> list[QuestionForTopic]:
    """
    if topics are provided, then chapter is ignored
    """

    topic_list: list[Topics] = []

    if len(topics) > 0:
        generic_kit = RAGtoolkit()
        metadatas = generic_kit.get_all_metadata()
        res = await client.chat(
            model="deepseek-r1",
            messages=[
                {
                    "role": "user",
                    "content": topics_filteration_prompt.format(metadatas, topics),
                }
            ],
            format=TopicList.model_json_schema(),
            options={"num_ctx": 8192, "temperature": 0},
            keep_alive=0,
        )

        topic_list.extend(TopicList.model_validate_json(res.message.content).topic_list)

    else:
        chapter_specific_kit = RAGtoolkit(chapter_num)
        raw_topics = chapter_specific_kit.get_chapter_metadata()["topics"]
        topics = convert_string_to_list(raw_topics)

        topic_list.append(
            Topics(
                file_name=chapter_specific_kit.chapter_name,
                chapter_num=chapter_num,
                topics=topics,
            )
        )

    questions = []
    for consolidated_topics in topic_list:
        topics = consolidated_topics.topics
        chapter_num = consolidated_topics.chapter_num
        chapter_toolkit = RAGtoolkit(chapter_num=chapter_num)

        max_retries = 3
        execution_timeout = 60
        for topic in topics:
            chunks = chapter_toolkit.query_docs(topic, n_results=10)["documents"][0]
            if len(chunks) > 0:
                for attempt in range(1, max_retries + 1):
                    task = asyncio.create_task(get_questions(chunks, topic))
                    try:
                        output = await asyncio.wait_for(task, timeout=execution_timeout)
                        print(output)
                        questions.append(output)
                        break
                    except asyncio.TimeoutError:
                        task.cancel()
                        if attempt < max_retries:
                            execution_timeout += 15
                            print(f"Attempt {attempt} timed out. Retrying...")
                        else:
                            print(
                                f"Attempt {attempt} timed out. Maximum retries reached; giving up."
                            )
                            exit()
            else:
                continue

    return questions


async def main():
    # ans = await DocumentQnA("What is a transition function?")
    # print(ans)
    """
    output:
    question='What is a transition function?'
    answer="The transition function describes how an automaton changes its state based on input.
    It's denoted by δ(q, a) = p, where q and p are states and a is an input symbol."
    """
    topics = ["FSA", "Language", "FSM", "Alphabet", "DFA", "NFA"]
    await QuestionsGeneration(topics, chapter_num=0)


if __name__ == "__main__":
    asyncio.run(main())


"""
2. Questions generation pipeline -> used to generate questions on per chapter basis, or per topic basis
3. Answers generation pipeline -> used to generate answers for each of the question
"""
