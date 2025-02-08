import os
import re
import uuid
import warnings
import torch
import numpy as np
from scipy.sparse import csr_matrix

from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)
from adapters import AutoAdapterModel
from semantic_text_splitter import TextSplitter
from transformers.utils import logging


import ollama
from pydantic import BaseModel

import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pgvector.psycopg2 import register_vector

# Set logging and warnings
logging.set_verbosity(40)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure cache directory exists
os.makedirs("cache", exist_ok=True)

# Load environment variables
load_dotenv()

# Connection parameters from environment variables
DB_USER = os.getenv("user")
DB_PASSWORD = os.getenv("password")
DB_HOST = os.getenv("host")
DB_PORT = os.getenv("DB_PORT", "5432")

ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

client = ollama.Client(host=ollama_url)


def get_conn_str(database: str) -> str:
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{database}"


splitter = TextSplitter(overlap=True, capacity=500, trim=True)

dense_tokenizer = AutoTokenizer.from_pretrained(
    "allenai/specter2_base", cache_dir="cache"
)
dense_model = AutoAdapterModel.from_pretrained(
    "allenai/specter2_base", cache_dir="cache"
)
dense_model.load_adapter(
    "allenai/specter2", source="hf", load_as="proximity", set_active=True
)

sparse_tokenizer = AutoTokenizer.from_pretrained(
    "naver/splade-cocondenser-ensembledistil", cache_dir="cache"
)
sparse_model = AutoModelForMaskedLM.from_pretrained(
    "naver/splade-cocondenser-ensembledistil", cache_dir="cache"
)

rerank_tokenizer = AutoTokenizer.from_pretrained(
    "cross-encoder/msmarco-MiniLM-L12-en-de-v1", cache_dir="cache"
)
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/msmarco-MiniLM-L12-en-de-v1", cache_dir="cache"
)

summary_generation_prompt = """
Using the following page info, present inside the page tags,
generate a 3 line summary and 3 tags for the page's content

<page>
{}
</page>
"""

summary_semantic_system = """
You are a content generation bot,
your role is to look at strings of text provided to you,
and chain them, such that the content makes sense semantically,
and allow content to be atleast 2 paragraphs long
"""


class SummaryAndTags(BaseModel):
    summary: str
    tags: list[str]


class CombinedSummary(BaseModel):
    summary: str


class RAGtoolkit:
    def __init__(self, chapter_num: int = 0, page_num: int = 1, topic: str = "general"):
        self.db_name = topic  # Target database name

        # Connect to default postgres DB first
        default_conn = psycopg2.connect(get_conn_str("postgres"), connect_timeout=1)
        default_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = default_conn.cursor()

        # Check if the target database exists
        cursor.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s;", (self.db_name,)
        )
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(f'CREATE DATABASE "{self.db_name}"')

        cursor.close()
        default_conn.close()

        # Now connect to the target database
        self.conn = psycopg2.connect(get_conn_str(self.db_name), connect_timeout=1)
        self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self.cursor = self.conn.cursor()

        # Ensure the vector extension exists
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create embeddings table for dense vectors
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_dense (
                id UUID PRIMARY KEY,
                chapter_num INT,
                page_num INT,
                content TEXT,
                embedding VECTOR(768)
            );
            """
        )

        # Create embeddings table for sparse vectors
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_sparse (
                id UUID PRIMARY KEY,
                chapter_num INT,
                page_num INT,
                content TEXT,
                sparse_embedding sparsevec(30522)
            );
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS topic_meta_data (
                id UUID PRIMARY KEY,
                chapter_num INT,
                page_num INT,
                summary TEXT,
                tags TEXT
            );
            """
        )

        # Register vector type with psycopg2
        register_vector(self.conn, arrays=True)

        # Object tagging attributes
        self.chapter_name = f"chapter_{chapter_num}"
        self.chapter_num = chapter_num
        self.page_number = page_num

    def _generate_dense_embeddings(self, inputs: list[str]):
        inputs = dense_tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )
        output = dense_model(**inputs)
        # take the first token in the batch as the embedding
        embeddings = output.last_hidden_state[:, 0, :].tolist()
        return embeddings

    def _generate_sparse_embeddings(self, docs):
        inputs = sparse_tokenizer(
            docs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            outputs = sparse_model(**inputs)
            logits = outputs.logits
            # Apply activation function (e.g., ReLU) to obtain sparse representations
            activations = torch.relu(logits)
            # Convert to sparse format
            sparse_embeddings = []
            for activation in activations:
                indices = activation.nonzero(as_tuple=True)[1].cpu().numpy()
                values = activation[activation != 0].cpu().numpy()
                sparse_vector = csr_matrix(
                    (values, (np.zeros_like(indices), indices)), shape=(1, 30522)
                )
                sparse_embeddings.append(sparse_vector)
        return sparse_embeddings

    def generate_chunks(self, doc: str) -> list[str]:
        """
        Splits the document into chunks such that any <table>...</table> or
        <image>...</image> block remains intact as a single chunk, while the
        rest of the content is handled by self.splitter.
        """
        # This pattern captures any <table>...</table> or <image>...</image> section
        pattern = r"(<table>.*?</table>|<image>.*?</image>)"
        # Split the document by these patterns, retaining the delimiter in the result
        segments = re.split(pattern, doc, flags=re.DOTALL)

        final_chunks: list[str] = []
        for segment in segments:
            # If it's a table or image block, keep it as one chunk
            if segment.strip().startswith("<table>") or segment.strip().startswith(
                "<image>"
            ):
                final_chunks.append(segment)
            else:
                # Otherwise, chunk the text via the default splitter
                # (assuming self.splitter.chunks(...) returns list[str])
                sub_chunks = splitter.chunks(segment)
                final_chunks.extend(sub_chunks)

        return final_chunks

    def generate_summary(self, page_content: str) -> SummaryAndTags:
        res = client.chat(
            model="huihui_ai/deepseek-r1-abliterated",
            messages=[
                {
                    "role": "user",
                    "content": summary_generation_prompt.format(page_content),
                }
            ],
            format=SummaryAndTags.model_json_schema(),
            options={"num_ctx": 16384, "temperature": 0.2},
            keep_alive=0,
        )
        return SummaryAndTags.model_validate_json(res.message.content)

    def add_meta(self, summary: str, tags: str, chapter_num: int, page_num: int):
        query = """
            INSERT INTO topic_meta_data (id, chapter_num, page_num, summary, tags)
            VALUES (%s, %s, %s, %s, %s);
        """

        data = (str(uuid.uuid1()), chapter_num, page_num, summary, tags)
        self.cursor.execute(query, data)

    def add_docs(self, docs: list[str]):
        # Generate embeddings for the documents
        dense_embeddings = self._generate_dense_embeddings(docs)
        sparse_embeddings = self._generate_sparse_embeddings(docs)

        # Generate unique UUIDs for each document
        doc_ids = [str(uuid.uuid1()) for _ in range(len(docs))]

        # Prepare data for batch insertion
        dense_data = [
            (doc_id, self.chapter_num, self.page_number, content, embedding)
            for doc_id, content, embedding in zip(doc_ids, docs, dense_embeddings)
        ]

        sparse_data = [
            (
                doc_id,
                self.chapter_num,
                self.page_number,
                content,
                sparse_embedding.toarray().tolist()[0],
            )
            for doc_id, content, sparse_embedding in zip(
                doc_ids, docs, sparse_embeddings
            )
        ]

        # SQL query for insertion
        dense_sql = f"""
            INSERT INTO embeddings_dense (id, chapter_num, page_num, content, embedding)
            VALUES %s
        """

        sparse_sql = """
            INSERT INTO embeddings_sparse (id, chapter_num, page_num, content, sparse_embedding)
            VALUES %s
        """

        # Execute batch insertion
        execute_values(self.cursor, dense_sql, dense_data)
        execute_values(self.cursor, sparse_sql, sparse_data)

    def _query_docs_dense(self, query: str, n_results: int = 20):
        # Generate the embedding for the query
        query_embedding = self._generate_dense_embeddings([query])

        # SQL query to find the most similar documents
        sql = """
            SELECT content
            FROM embeddings_dense
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        # Execute the query
        self.cursor.execute(sql, (query_embedding[0], n_results))

        # Fetch the results
        results = self.cursor.fetchall()

        # Process and return the results as needed
        return results

    def _query_docs_sparse(self, query: str, n_results: int = 20):
        # Generate the sparse embedding for the query
        query_embedding = (
            self._generate_sparse_embeddings([query])[0].toarray().tolist()[0]
        )

        # SQL query to find the most similar documents
        sql = """
            SELECT content
            FROM embeddings_sparse
            ORDER BY sparse_embedding <=> %s::sparsevec
            LIMIT %s;
        """

        # Execute the query
        self.cursor.execute(sql, (query_embedding, n_results))

        # Fetch the results
        results = self.cursor.fetchall()

        # Process and return the results as needed
        return results

    def query_docs(self, query: str, top_n: int = 20):
        dense_results = self._query_docs_dense(query)
        sparse_results = self._query_docs_sparse(query)

        data = list(result[0] for result in dense_results) + list(
            result[0] for result in sparse_results
        )

        combined_inputs = [f"{query} [SEP] {doc}" for doc in data]

        embeddings = rerank_tokenizer(
            combined_inputs,
            return_tensors="pt",  # Ensure tensors are returned
            padding=True,  # Optional: pad sequences to the same length
            truncation=True,  # Optional: truncate sequences to a maximum length
        )

        rerank_model.eval()
        with torch.no_grad():
            scores = rerank_model(**embeddings).logits
            scores_list = scores.tolist()

            scored_data = list(zip(scores_list, data))

            # Sort the combined list by scores in descending order
            scored_data.sort(key=lambda x: x[0], reverse=True)

            # Extract the sorted data entries
            top_reranked_data = [item[1] for item in scored_data[:top_n]]
            return top_reranked_data

    def get_chapter_tags(self, chapter_num: int) -> list[str]:

        query = """
            SELECT tags FROM topic_meta_data WHERE chapter_num = %s;
        """
        self.cursor.execute(query, (chapter_num,))
        results = self.cursor.fetchall()

        unique_tags = set()

        for row in results:
            tags_str = row[0]  # The stored tags string
            tags_list = convert_string_to_list(tags_str)  # Convert string to list
            unique_tags.update(
                tag.lower() for tag in tags_list
            )  # Convert to lowercase before adding to set

        return list(unique_tags)

    def get_chapter_summary(self, chapter_num: int) -> str:
        query = """
            SELECT summary FROM topic_meta_data
            WHERE chapter_num = %s
            ORDER BY page_num ASC;
        """
        self.cursor.execute(query, (chapter_num,))
        results = self.cursor.fetchall()

        # Concatenate all summaries
        concatenated_summary = " ".join(
            f"""
\n-------Page {index}------\n
    {row[0]}
\n-------Page {index}------\n
            """
            for (index, row) in enumerate(results)
            if row[0]
        )

        res = client.chat(
            model="huihui_ai/deepseek-r1-abliterated",
            messages=[
                {"role": "system", "content": summary_semantic_system},
                {
                    "role": "user",
                    "content": concatenated_summary,
                },
            ],
            options={"temperature": 0.4, "num_ctx": 32768},
            keep_alive=0,
            format=CombinedSummary.model_json_schema(),
        )

        return CombinedSummary.model_validate_json(res.message.content).summary


import torch.nn.functional as F


def grade_statement_similarity(user_statement: str, actual_statement: str) -> float:
    """
    Evaluates the similarity between a user's answer and the actual answer
    using the cross-encoder reranker model. The function returns a similarity
    score (as a percentage from 0% to 100%).

    Args:
        user_statement (str): The statement provided by the user.
        actual_statement (str): The reference or actual statement.

    Returns:
        float: Similarity score as a percentage.
    """
    # Format the input as a query-document pair (using [SEP] as a separator)
    input_text = f"{user_statement} [SEP] {actual_statement}"

    # Tokenize the input; the model expects a single sequence containing both answers
    inputs = rerank_tokenizer(
        input_text, return_tensors="pt", truncation=True, padding=True
    )

    # Ensure the model is in evaluation mode
    rerank_model.eval()

    # Compute logits without gradient computation
    with torch.no_grad():
        logits = rerank_model(**inputs).logits  # Expected shape: [1, num_labels]

    # If the model returns only one logit (i.e. num_labels==1), use sigmoid;
    # otherwise, use softmax to extract the probability for the positive class.
    if logits.size(1) == 1:
        # Use sigmoid for single-output scenario
        prob = torch.sigmoid(logits)[0][0].item()
    else:
        # Use softmax for standard binary classification (positive class at index 1)
        prob = F.softmax(logits, dim=-1)[0][1].item()

    # Convert the probability (0 to 1) into a percentage (0% to 100%)
    return prob * 100


# cause metadata won't accept list data, it has to be str
def convert_string_to_list(list_str: str) -> list[str]:
    """
    Convert a string that looks like a list into an actual list.

    Args:
        list_str (str): The input string that contains a list.

    Returns:
        list[str]: A list of strings if the input string is valid, otherwise an empty list.
    """

    # Use regular expression to find the list
    match = re.search(r"\[(.*)\]", list_str)
    if match:
        content = match.group(1).strip()
        # Split the content by commas and strip quotes from each item
        actual_list = [
            item.strip("'\"")
            for item in content.replace('"', "").replace("'", "").split(",")
        ]
        actual_list = [item.strip() for item in actual_list]

        return actual_list
    else:
        print("No list found in the string")
        return []
