# **RAG Toolkit Usage Markdown**

### Overview

The RAG Toolkit is a comprehensive tool for consuming documentation, generating questions on docs, evaluating answers on doc, and asking questions to users.

### Subcommands

#### 1. `read`

- **Usage:** `python main.py read --for <topic> --from <file_path> --chapter_num <chapter_num> --structure <doc_structure> --manual_terminate <title>`
- **Description:** Reads data from a file and extracts relevant information.
- **Flags:**
  - `--for`: Topic name
  - `--from`: Path to input file
  - `--chapter_num`: Chapter number
  - `--structure`: Document structure (default: "research/default")
  - `--manual_terminate`: Title for manual termination

#### 2. `generate`

- **Usage:** `python main.py generate --for <topic> --chapter_num <chapter_num> --sub-topics <tags>`
- **Description:** Generates questions for a given topic and chapter.
- **Flags:**
  - `--for`: Topic name
  - `--chapter_num`: Chapter number
  - `--sub-topics`: Tags associated with the topic

#### 3. `summarize`

- **Usage:** `python main.py summarize --for <topic> --chapter_num <chapter_num>`
- **Description:** Summarizes chapter content.
- **Flags:**
  - `--for`: Topic name
  - `--chapter_num`: Chapter number

#### 4. `evaluate`

- **Usage:** `python main.py evaluate --for <topic> --question <question_string> --answer <user_answer>`
- **Description:** Evaluates a question-answer pair.
- **Flags:**
  - `--for`: Topic name
  - `--question`: Question string
  - `--answer`: User's answer to the question

#### 5. `answer`

- **Usage:** `python main.py answer --for <topic> --question <question_string>`
- **Description:** Answers a user's question.
- **Flags:**
  - `--for`: Topic name
  - `--question`: Question to answer for the user

### Example Usage

```bash
# Read data from a file and extract relevant information
python main.py read --for python --from example.pdf --chapter_num 1 --structure research/default --manual_terminate Python Documentation

# Generate questions for a given topic and chapter
python main.py generate --for python --chapter_num 1 --sub-topics programming, data structures

# Summarize chapter content
python main.py summarize --for python --chapter_num 1

# Evaluate a question-answer pair
python main.py evaluate --for python --question What is Python? --answer I love Python!

# Answer a user's question
python main.py answer --for python --question What is the meaning of life?
```

### Flags and Examples

| Flag                 | Description                                                         | Example                |
| -------------------- | ------------------------------------------------------------------- | ---------------------- |
| `--for`              | Topic name                                                          | `python`               |
| `--from`             | Path to input file (file needs to be stored in /files dir)          | `/files/example.pdf`   |
| `--chapter_num`      | Chapter number                                                      | `1`                    |
| `--structure`        | Document Structure (`research/default`)                             | `research/default`     |
| `--manual_terminate` | Title String on which manual termination, of parsing should be done | `Python Documentation` |

```bash
# Generate questions for a given topic and chapter
python main.py generate --for python --chapter_num 1 --sub-topics "programming+data structures+algorithms"

# Summarize chapter content
python main.py summarize --for python --chapter_num 1

# Evaluate a question-answer pair
python main.py evaluate --for python --question "What is Python?" --answer "I love Python!"

# Answer a user's question
python main.py answer --for python --question "What is the meaning of life?"
```
