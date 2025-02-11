# **RAG Toolkit Usage Markdown**

### Overview

The RAG Toolkit is a comprehensive tool for consuming documentation, generating questions on docs, evaluating answers on doc, and asking questions to users.

### Subcommands

#### 1. `read`

- **Usage:** `python src/main.py read --for <topic> --from <file_path> --chapter_num <chapter_num> --structure <doc_structure> --manual_terminate <title>`
- **Description:** Reads data from a file and extracts relevant information.
- **Flags:**
  - `--for`: Topic name
  - `--from`: Path to input file
  - `--chapter_num`: Chapter number
  - `--structure`: Document structure (default: "research/default")
  - `--manual_terminate`: Title string content at which the parser should manually terminate

NOTE: the extracted documents are saved in `/outputs` directory, organized by the topic name

#### 2. `generate`

- **Usage:** `python src/main.py generate --for <topic> --chapter_num <chapter_num> --sub-topics <tags>`
- **Description:** Generates questions for a given topic and chapter.
- **Flags:**
  - `--for`: Topic name
  - `--chapter_num`: Chapter number
  - `--sub-topics`: Tags associated with the topic

NOTE: the resulting questions are stored in `/generated` directory

#### 3. `summarize`

- **Usage:** `python src/main.py summarize --for <topic> --chapter_num <chapter_num>`
- **Description:** Summarizes chapter content.
- **Flags:**
  - `--for`: Topic name
  - `--chapter_num`: Chapter number

#### 4. `evaluate`

- **Usage:** `python src/main.py evaluate --for <topic> --question <question_string> --answer <user_answer>`
- **Description:** Evaluates a question-answer pair.
- **Flags:**
  - `--for`: Topic name
  - `--question`: Question string
  - `--answer`: User's answer to the question

#### 5. `answer`

- **Usage:** `python src/main.py answer --for <topic> --question <question_string>`
- **Description:** Answers a user's question.
- **Flags:**
  - `--for`: Topic name
  - `--question`: Question to answer for the user

NOTE: answers once generated, are cached for each topic, in the file `generated/topic_name/QnA.json`

### Example Usage

```bash
# Read data from a file and extract relevant information
python src/main.py read --for python --from example.pdf --chapter_num 1 --structure research --manual_terminate "Python Documentation"

# Generate questions for a given topic and chapter
python src/main.py generate --for python --chapter_num 1 --sub-topics "arrays+loops"

# Summarize chapter content
python src/main.py summarize --for python --chapter_num 1

# Evaluate a question-answer pair
python src/main.py evaluate --for python --question "What is Python?" --answer "I love Python!"

# Answer a user's question
python src/main.py answer --for python --question "What is the meaning of life?"
```

### Flags and Examples

| Flag                 | Description                                                           | Example                                  |
| -------------------- | --------------------------------------------------------------------- | ---------------------------------------- |
| `--for`              | Topic name                                                            | `python`                                 |
| `--from`             | Path to input file (file needs to be stored in /files dir)            | `/files/example.pdf`                     |
| `--chapter_num`      | Chapter number                                                        | `1`                                      |
| `--structure`        | Document Structure (`research/default`)                               | `research/default`                       |
| `--manual_terminate` | Title String on which manual termination, of parsing should be done   | `Python Documentation`                   |
| `--sub-topics`       | Actual Topics in the Subject, that need to be targeted                | `programming+data structures+algorithms` |
| `--question`         | The Question String, used for either evaluation, or Answer Generation | `What is Python?`                        |
| `--answer`           | The Answer String, used for answer evaluation                         | `I love Python!`                         |

```bash
# Generate questions for a given topic and chapter
python src/main.py generate --for python --chapter_num 1 --sub-topics "programming+data structures+algorithms"

# Summarize chapter content
python src/main.py summarize --for python --chapter_num 1

# Evaluate a question-answer pair
python src/main.py evaluate --for python --question "What is Python?" --answer "I love Python!"

# Answer a user's question
python src/main.py answer --for python --question "What is the meaning of life?"
```
