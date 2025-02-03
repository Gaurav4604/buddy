from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from semantic_text_splitter import TextSplitter
import os
from os import listdir
from os.path import isfile, join


ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
import ollama
import uuid
from pydantic import BaseModel
import re
from transformers.utils import logging
import warnings

logging.set_verbosity(40)
warnings.filterwarnings("ignore", category=FutureWarning)


from dotenv import load_dotenv

load_dotenv()

meta_generation_prompt = """
You are a summary, tags and topics generator,
your role is to look at the document provided,
and generate category tags associated to it
along with all topics present in it and a 5 line summary,
you should limit the number of tags and topics to 3.
"""

summary_merge_prompt = """
You are a summary merge manager,
your role is to look at multi-page summaries,
and merge them into a single summary, and associate
5 tags with the same.
"""

os.makedirs("cache", exist_ok=True)


class TagsTopicsAndSummaryModel(BaseModel):
    tags: list[str]
    topics: list[str]
    summary: str


splitter = TextSplitter(overlap=True, capacity=500, trim=True)
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base", cache_dir="cache")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base", cache_dir="cache")
model.load_adapter(
    "allenai/specter2", source="hf", load_as="proximity", set_active=True
)


def generate_embeddings(inputs: list[str]):
    inputs = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=512,
    )
    output = model(**inputs)
    # take the first token in the batch as the embedding
    embeddings = output.last_hidden_state[:, 0, :].tolist()
    return embeddings


def generate_chunks(doc: str) -> list[str]:
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


import psycopg2

conn = psycopg2.connect(
    database=os.getenv("database"),
    user=os.getenv("postgres"),
    password=os.getenv("password"),
    host=os.getenv("host"),
    port=os.getenv("port"),
    connect_timeout=1,
)

cursor = conn.cursor()
cursor.execute(
    """
    SELECT * from items
    ORDER BY embedding <=> '[1, 1, 1]'
    """
)

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
