# Buddy

A Retrieval Augmented Generation Based toolkit, to help students studying STEM subjects offload tasks such as

1. Topic and Chapter targeted Question Generation for their subject topics
2. Chain of Question based complex and reasoning rich question answering
3. User Answer Evalution and Grading
4. Semantically Sensible Summary Generation for Chapters

## Setup and How to Use

### Installation

#### Using Docker _(recommended)_

Make sure to have docker, along with docker-compose setup on your system.

```bash
# clone the repo
git clone repo-name

# using docker
docker-compose build

docker-compose up
```

### Usage

-- to add docker based usage, once docker image is setup to handle main.py files --

## Motivation and Inner Workings

### What happens and Why?

The application works in 3 very distinct steps

1.  Document Processing:<br/>

    1. Document Content Chunk Extraction: The document PDF file is read, and on each page, using a [YOLO model for extracting document layout](https://github.com/opendatalab/DocLayout-YOLO), each page is divided into chunks. Based on the content flow specified by the command line args (research/default) the chunks are re-ordered to make sense semantically. Next, depending on the data type _(image, text, formula, tables)_
    2. Chunk Content Extraction: The data present in each chunk is then extracted out using a combination of LLMs and traditional OCR (vision llm and OCR for data extraction, text llms for data filtering and formatting). **This approach was much more useful and worked better than using vision llm directly for content extraction, since this retained the exact content data most of the time.**
    3. Chunk Re-stitching: All chunks for each page are then re-ordered into an array, and subsequently stored as `.txt` file for each page. This approach allows me to allow user to validate the contents of extraction, and edit them if needed _(considering I'm using LLMs, and they tend to hallucinate sometimes, it happens rarely, but it does)_

2.  Embeddings Generation, Storage and Retrieval _(defined as RAGToolkit)_: <br/>

    1. Embedding Generation: For **each topic**, a new database in postgres SQL, is created, with 3 tables:

       - sparse_embeddings (using naver/splade-cocondenser-ensembledistil)
       - dense_embeddings (using allenai/specter2) <- this embeddings works best on scientific data
       - topic_meta_data (to store a summary of each page, along with tags containing all topics present on the same page, along with chapter_num)

       A file parser, chunks my `.txt` files, and generates embeddings, along with an LLM based summary of the page, finally storing all of the same to the database

    2. RAG Context Document Preparation: The best part about using vector-databases, is its ability to find query relevant answers, in my current case, I use a process of: <br/>

       1. Query the dense table
       2. Query the sparse table
       3. using reranking, sort and filter query relevant results
       4. return these results, for context generation for the LLM

    3. Semantically Sound Summary Generation: Using an LLM, to stitch together, each page's summary, to form a chapter-wise summary on demand, for easily accessible synopsis of the chapter as and when required.

    4. Sentence Similarity Evaluation: Since, re-ranking is nothing but evaluation of how similar two statements are, I used the same, to build out a function that returns a similarity percentage between query and its target, which can be used for answer evaluation.

       <br/>_(I used this approach since it was a very easy to use postgres as a vectordb, which also allowed to be query using traditional SQL queries. I'd initially tried chromadb, but it fought with be quite a bit, and seemed very well oriented towards langchain based workflows, once I did get it working, it did not allow me to finetune properly for sparse vectors, which was final nail in the cofine to use postgres)_

3.  LLM based RAG tools: Ofcourse, the best part about an LLM-centric system is the sweet sweet answer generation capabilities, and I've tried to leverage them as best possible for my use-case:

    1. Chain of Question Reasoning: Honestly this is the gem of my project, targetted towards thinking like a human, in decomposing a complex question, using a reasoning model like `deepseek-r1`, into multiple atomic questions, that each target a particular section of the question. Here's how this works:

       1. Complex Query is broken into simpler atomic queries, that are sequentially relevant, forming a chain of questions, that ultimately try to answer the initial question. `Q -> [q1, q2, q3]`
       2. Each query is answered sequentially, which forms additional context for the next question.

          ```
          answer_1 = Fn(q1)
          answer_2 = Fn(q2, [(q1, answer_1)])
          answer_3 = Fn(q3, [(q1, answer_1), (q2, answer_2)])
          ```

          Thus beautifully decomposing the initial question into simpler easier questions to answer.

       3. Once this context is formed, by answering all of the simpler queries, the model, asks itself the original question again, with these atomic answers as reference, to ground its answer, and thus provides an accurate answer to user's question

       _(ofcourse, the atomic questions are answered, by using context provided using the vectordb, via the RAGToolkit)_

    2. Questions Generation: Using simple prompt template, along with the RAGToolkit, I get all topics (either user-defined, or chapter_defined), to generate context, which is then used by the LLM along with the prompt template, to generate questions for the user, stored in as `.json` file for ease of reading and access.

## Motivation

It started out with building an LLM based Document Parser, since the only alternative which was available
to consume highly dense and multi-format rich documents, was [Llama-Parse](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/) which was paid, and also required me to send my data over the internet _(making it not so private 😅)_

Later on, during my exams, I also suffered from a lack of Question Banks, that targeted my known weak topics for each of the subjects, all of which prompted me to build my own Toolkit, to find solutions to these problems.

I want to use AI to solve realworld problems in domains, where it augments the user's skillset, rather than taking over user's tasks.

RAG, being a very rapidly researched domain, seemed like a good solution fit for my use-case, thus leading to this toolkit.
