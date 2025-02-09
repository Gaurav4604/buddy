# Buddy

A Retrieval Augmented Generation Based toolkit, to help students studying STEM subjects offload tasks such as

1. Question Generation for Subjects
2. Question Answering and Answer Evalution
3. Summary Generation for Chapters

## Project Description

### What happens and Why?

The application works in 3 very distinct steps

1.  Document Processing:<br/>
    1. Document Content Chunk Extraction: The document PDF file is read, and on each page, using a [YOLO model for extracting document layout](https://github.com/opendatalab/DocLayout-YOLO), each page is divided into chunks. Based on the content flow specified by the command line args (research/default) the chunks are re-ordered to make sense semantically. Next, depending on the data type _(image, text, formula, tables)_

## Motivation

It started out with building an LLM based Document Parser, since the only alternative which was available
to consume highly dense and multi-format rich documents, was [Llama-Parse](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/) which was paid, and also required me to send my data over the internet _(making it not so private 😅)_

Later on, during my exams, I also suffered from a lack of Question Banks, that targeted my known weak topics for each of the subjects, all of which prompted me to build my own Toolkit, to find solutions to these problems.

RAG, being a very rapidly researched domain, seemed like a good solution fit for my use-case, thus leading to this toolkit.
