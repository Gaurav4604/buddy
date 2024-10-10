# Motivation

Document preprocessing is required for RAG (Retrieval-Augmented Generation) chain processing, and documents should be properly formatted, as improper formatting will result in a garbage-in, garbage-out scenario. Current OCR methods struggle with extracting documents that contain both text and math symbols, often misinterpreting mathematical expressions or losing important text data. The more accurate solutions are cloud-based, expensive, and locked behind paywalls. To address this, I’ve used LLaMA 3.2 (3B) with OCR and specialized math extraction, merging the results for more accurate document processing. This setup:

1. Converts PDFs to text,
2. Processes math into LaTeX format and merges the data,
3. Reformats it to display math symbols as plaintext,

all within a self-hosted Docker image, providing an affordable and robust solution.

## Tools and Requirements for Installation

- Docker:
    The entire setup relies on Docker containers to run the services. You need Docker installed on your system.

- Docker Compose:
    To orchestrate multiple containers, you need Docker Compose.

## How to Use

- place your file, name it **file.pdf** in the root directory and run the following command: `docker-compose up --build`

## Outputs

- **/tex** contains the converted LaTeX files, indexed based on page number.
- **/txt** contains the converted plain text files, indexed based on page number.

## Future Goals

1. Add diagram processing options (using Multi-Modal LLMs).
2. Add document tagging, to add meta-data information based on document contents.
3. Implement RAG (Retrieval-Augmented Generation) chain processing, for QnA generation.
