import ollama
from pydantic import BaseModel
from .utils import RAGtoolkit
import os
import asyncio


ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

client = ollama.AsyncClient(host=ollama_url)


class QuestionDecomposed(BaseModel):
    original_question: str
    sub_questions: list[str]
    category: str


question_decomposition_system = """
You are incharge of breaking the given question down into simpler sub-questions,
which can then be used to answer the original question again
"""

question_decomposition_examples = """
I will provide you with some example questions and its simpler sub-questions, later I will
also provide you with the question, I want you to de-compose into simpler sub-questions

Types:
    1. Comparison:
        Examples:
        - Question: What are the differences between A and B?
        - Sub-Questions generated:
                1. What is the definition of A?
                2. What is the definition of B?
        - Question: In what distinct ways do A and B vary?
        - Sub-Questions generated:
                1. How is A fundamentally described?
                2. How is B fundamentally described?
    
    2. Contrast:
        - Question: In what ways do A and B differ in their inherent characteristics?
        - Sub-Questions generated:
            1. What are the essential features of A?
            2. What are the essential features of B?
        - Question: How can the intrinsic qualities of A be distinguished from those of B?
        - Sub-Questions generated:
            1. What intrinsic qualities define A?
            2. Which qualities clearly separate A from B?

    3. Synthesis:
        - Question: How can elements from A, B, and C be combined to create a new integrated model?
        - Sub-Questions generated:
            1. What are the key components or principles of A?
            2. What are the key components or principles of B?
            3. What are the key components or principles of C?
        - Question: What approach can merge the components of A, B, and C into a unified framework?
        - Sub-Questions generated:
            1. What fundamental aspects constitute A?
            2. What fundamental aspects constitute B?
            3. What fundamental aspects constitute C?
    
    4. Analysis:
        - Question: How can A be broken down into its constituent parts, and what is the role of each part?
        - Sub-Questions generated:
            1. What are the individual components that make up A?
            2. What function does each component serve within A?
            3. How do these components interact to form the overall structure of A?
        - Question: In what manner can A be dissected into its individual segments, and what function does each segment perform?
        - Sub-Questions generated:
            1. What segments comprise A?
            2. What is the role of each segment?
            3. How do these segments combine to establish A's complete structure?

    5. Evaluation:
        - Question: How effective is A in achieving its intended purpose, and what are its strengths and limitations?
        - Sub-Questions generated:
            1. What criteria can be used to assess the performance of A?
            2. How does A perform according to these criteria?
            3. What strengths does A exhibit?
            4. What limitations or weaknesses are evident in A?
        - Question: To what extent does A fulfill its goals, and what are its notable advantages and drawbacks?
        - Sub-Questions generated:
            1. What benchmarks determine A's success?
            2. How well does A meet these benchmarks?
            3. What are the major advantages of A?
            4. What are the key drawbacks of A?

    6. Application:
        - Question: How can the concepts of A be applied to solve challenges presented by B?
        - Sub-Questions generated:
            1. What are the fundamental principles or techniques underlying A?
            2. What specific challenges or requirements does B present?
            3. How can the principles of A be mapped to address the challenges of B?
            4. What potential issues might arise when applying A to the context of B?
        - Question: In what ways can the principles of A be utilized to tackle the problems encountered in B?
        - Sub-Questions generated:
            1. What are the core principles of A?
            2. What problems or challenges does B pose?
            3. How can these principles be effectively adapted to resolve the challenges of B?
            4. What obstacles could emerge during this adaptation?

    7. Definition:
        - Question: What is the definition of A?
        - Sub-Questions generated:
                1. What are the essential characteristics that define A?
                2. How is A fundamentally described within its context?
        - Question: How can A be defined in relation to B?
        - Sub-Questions generated:
                1. What is the core meaning of A?
                2. How does the relationship with B shape the definition of A?

    8. Explanation:
        - Question: How can the workings of A be explained?
        - Sub-Questions generated:
                1. What processes underlie the operation of A?
                2. What are the key mechanisms that drive A?
        - Question: Can you explain how A functions within the framework of B?
        - Sub-Questions generated:
                1. What role does A play in the functioning of B?
                2. How do the interactions between A and B facilitate overall performance?

    9. Justification:
        - Question: Why is A considered essential for B?
        - Sub-Questions generated:
                1. What evidence supports the critical role of A in B?
                2. How does A contribute to the overall success or efficiency of B?
        - Question: What are the reasons for implementing A in the context of B?
        - Sub-Questions generated:
                1. What benefits does A offer in this scenario?
                2. How do these benefits outweigh any potential drawbacks?
"""

question_decomposition_template = """
using the examples mentioned, decompose the question enclosed in tags,
into simpler sub-questions, and tag it into one of the categories as defined above

<question>
{}
</question>
"""


async def decompose_question(question: str) -> QuestionDecomposed:
    res = await client.chat(
        model="huihui_ai/deepseek-r1-abliterated",
        messages=[
            {
                "role": "system",
                "content": question_decomposition_system,
            },
            {
                "role": "user",
                "content": question_decomposition_examples,
            },
            {
                "role": "user",
                "content": question_decomposition_template.format(question),
            },
        ],
        format=QuestionDecomposed.model_json_schema(),
        options={"num_ctx": 32768, "temperature": 0.2},
        keep_alive=0,
    )
    return QuestionDecomposed.model_validate_json(res.message.content)


class QuestionAnswer(BaseModel):
    question: str
    answer: str


question_answer_system = """
You're a Retrieval Augmented Question Answering Bot,
that answers user's question, based on the reference text provided,
Your task is to do reasoning, on the reference text,
in accordance to the user's question,
and provide a 5-6 line answer to the same
"""

question_answer_template = """
the following, enclosed in tags is the reference text

<reference>
{}
</reference>

answer this question, using the above reference
<question>
{}
</question>
"""

question_answer_with_question_chain_template = """
the following, enclosed in tags is the reference text

<reference>
{}
</reference>

the following, is answers to questions related to this subject,
to be used as reference in answering the current question

<previous-question-answer>
{}
</previous-question-answer>

answer this question, using the above references
and previous questions and answers
<main-question>
{}
</main-question>
"""


async def answer_atomic_question(
    question: str, topic: str = "general", question_chain: list[str] = []
) -> QuestionAnswer:
    kit = RAGtoolkit(topic=topic)
    docs = kit.query_docs(query=question)

    prompt = ""

    if len(question_chain) > 0:
        prompt = question_answer_with_question_chain_template.format(
            "\n----reference----\n".join(docs),
            "\n------previous_questions-----\n".join(question_chain),
            question,
        )
    else:
        prompt = question_answer_template.format(
            "\n----reference----\n".join(docs),
            question,
        )

    res = await client.chat(
        model="huihui_ai/deepseek-r1-abliterated",
        messages=[
            {"role": "system", "content": question_answer_system},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        format=QuestionAnswer.model_json_schema(),
        options={"num_ctx": 32768, "temperature": 0},
        keep_alive=0,
    )

    return QuestionAnswer.model_validate_json(res.message.content)


async def answer_composite_question(
    question: str, answers: list[QuestionAnswer]
) -> QuestionAnswer:

    reference_text = ""

    for answer in answers:
        reference_text += f"""
        ----reference----
        Question: {answer.question}
        Answer: {answer.answer}
        ----reference----
        """

    res = await client.chat(
        model="huihui_ai/deepseek-r1-abliterated",
        messages=[
            {"role": "system", "content": question_answer_system},
            {
                "role": "user",
                "content": question_answer_template.format(reference_text, question),
            },
        ],
        format=QuestionAnswer.model_json_schema(),
        options={"num_ctx": 32768, "temperature": 0},
        keep_alive=0,
    )

    return QuestionAnswer.model_validate_json(res.message.content)


class QuestionsFromTag(BaseModel):
    questions: list[str]
    tag: str


question_generation_system = """
You're a Retrieval Augmented Question Generation Bot,

Your role is look at the provided topic, 
along with the reference text associated with it,
and generate 2 questions from it:
 - 1st Question should be simple
 - 2nd Question should be complex and comprehensive
"""


question_generation_prompt = """
the following, enclosed in tags is the reference text

<reference>
{}
</reference>

generate questions for the following topic, using the above reference
<topic>
{}
</topic>
"""


async def generate_questions(tag: str, topic: str = "general") -> QuestionsFromTag:
    kit = RAGtoolkit(topic=topic)
    docs = kit.query_docs(query=tag)

    if len(docs):
        res = await client.chat(
            model="huihui_ai/deepseek-r1-abliterated",
            messages=[
                {"role": "system", "content": question_generation_system},
                {
                    "role": "user",
                    "content": question_generation_prompt.format(
                        "\n----reference----\n".join(docs), tag
                    ),
                },
            ],
            format=QuestionsFromTag.model_json_schema(),
            options={"num_ctx": 32768, "temperature": 0.2},
            keep_alive=0,
        )

        return QuestionsFromTag.model_validate_json(res.message.content)
    else:
        return QuestionsFromTag(questions=[], tag=tag)
