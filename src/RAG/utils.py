from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)
from adapters import AutoAdapterModel
from semantic_text_splitter import TextSplitter
import os
import torch
from scipy.sparse import csr_matrix
import numpy as np

from transformers.utils import logging
import warnings


import uuid
import re


logging.set_verbosity(40)
warnings.filterwarnings("ignore", category=FutureWarning)

os.makedirs("cache", exist_ok=True)

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

load_dotenv()

conn = psycopg2.connect(
    database=os.getenv("database"),
    user=os.getenv("postgres"),
    password=os.getenv("password"),
    host=os.getenv("host"),
    port=os.getenv("port"),
    connect_timeout=1,
)

conn.autocommit = True

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


class RAGtoolkit:
    def __init__(self, chapter_num: int = 0, page_num: int = 1):

        self.cursor = conn.cursor()

        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # each chapter should get its own table
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

        register_vector(conn, arrays=True)

        # tagging the object
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

        embeddings = rerank_tokenizer(
            data,
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
