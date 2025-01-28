from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from semantic_text_splitter import TextSplitter
from os import listdir
from os.path import isfile, join


import chromadb
from chromadb.config import Settings
from chromadb.api.types import QueryResult
from transformers.utils import logging
import warnings

import ollama
import uuid
from pydantic import BaseModel
import re
import os


logging.set_verbosity(40)
warnings.filterwarnings("ignore", category=FutureWarning)


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


class RAGtoolkit:
    def __init__(self, chapter_num: int = 0, page_num: int = 1):
        settings = Settings(is_persistent=True)
        self.splitter = TextSplitter(overlap=True, capacity=500, trim=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/specter2_base", cache_dir="cache"
        )
        self.model = AutoAdapterModel.from_pretrained(
            "allenai/specter2_base", cache_dir="cache"
        )
        self.model.load_adapter(
            "allenai/specter2", source="hf", load_as="proximity", set_active=True
        )
        self.client = chromadb.Client(settings=settings)
        # each chapter should get its own collection
        self._doc_collection = self.client.get_or_create_collection(
            name=f"chapter_{chapter_num}"
        )

        # this collection will be general to each chapter
        self._chapter_meta_collection = self.client.get_or_create_collection(
            name="chapters_meta"
        )
        self.ollama_client = ollama.Client()

        # tagging the object
        self.chapter_name = f"chapter_{chapter_num}"
        self.page_number = f"page_{page_num}"

    def _generate_embeddings(self, inputs: list[str]):
        inputs = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )
        output = self.model(**inputs)
        # take the first token in the batch as the embedding
        embeddings = output.last_hidden_state[:, 0, :].tolist()
        return embeddings

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
                sub_chunks = self.splitter.chunks(segment)
                final_chunks.extend(sub_chunks)

        return final_chunks

    def generate_meta(self, docs_dir: str) -> dict:
        tags = set()
        topics = set()
        files = [
            join(docs_dir, f) for f in listdir(docs_dir) if isfile(join(docs_dir, f))
        ]
        summarys = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                res = self.ollama_client.chat(
                    model="deepseek-r1",
                    messages=[
                        {"role": "system", "content": meta_generation_prompt},
                        {"role": "user", "content": f.read()},
                    ],
                    format=TagsTopicsAndSummaryModel.model_json_schema(),
                    options={"num_ctx": 8192},
                )
                page_level_meta = TagsTopicsAndSummaryModel.model_validate_json(
                    res["message"]["content"]
                )
                for tag in page_level_meta.tags:
                    tags.add(tag)
                for topic in page_level_meta.topics:
                    topics.add(topic)
                summarys.append(page_level_meta.summary)

        res = self.ollama_client.chat(
            model="deepseek-r1",
            messages=[
                {"role": "system", "content": summary_merge_prompt},
                {"role": "user", "content": summarys[0] + ", ".join(summarys[1:])},
            ],
            format=TagsTopicsAndSummaryModel.model_json_schema(),
            options={"num_ctx": 8192},
        )
        overall_summary = TagsTopicsAndSummaryModel.model_validate_json(
            res["message"]["content"]
        ).summary

        return {"tags": list(tags), "topics": list(topics), "summary": overall_summary}

    # chunks for the given chapter to be added
    def add_docs(self, docs: list[str]):
        self._doc_collection.upsert(
            documents=docs,
            metadatas=[
                {"file_name": f"{self.chapter_name}/{self.page_number}.txt"}
                for _ in range(len(docs))
            ],
            ids=[str(uuid.uuid1()) for _ in range(len(docs))],
            embeddings=self._generate_embeddings(docs),
        )

    def add_meta_data(self, tags: list[str], topics: list[str], summary: str):
        doc = "<TAGS> tags:" + " - ".join(tags) + "</TAGS>\n"
        doc = "<TOPICS> topics:" + " - ".join(topics) + "</TOPICS>\n"
        doc += summary

        self._chapter_meta_collection.upsert(
            documents=[doc],
            metadatas=[
                {
                    "file_name": self.chapter_name,
                    "tags": str(tags),
                    "topics": str(topics),
                }
            ],
            ids=[self.chapter_name],
            embeddings=self._generate_embeddings(doc),
        )

    def query_docs(self, query: str, n_results: int = 2) -> QueryResult:
        res = self._doc_collection.query(
            query_embeddings=self._generate_embeddings(query),
            query_texts=[query],
            n_results=n_results,
        )
        return res

    def query_metadata(self, query: str) -> QueryResult:
        res = self._chapter_meta_collection.query(
            query_embeddings=self._generate_embeddings(query), query_texts=[query]
        )
        return res

    def get_summary(self) -> str:
        res = self._chapter_meta_collection.query(
            query_texts=[""],
            where={"file_name": self.chapter_name},
            query_embeddings=self._generate_embeddings(""),
            n_results=1,
        )
        return res["documents"][0]

    def get_chapter_metadata(self) -> dict:
        res = self._chapter_meta_collection.query(
            query_texts=[""],
            where={"file_name": self.chapter_name},
            query_embeddings=self._generate_embeddings(""),
            n_results=1,
        )
        return res["metadatas"][0][0]

    def get_all_metadata(self) -> list:
        res = self._chapter_meta_collection.query(
            query_texts=[""],
            query_embeddings=self._generate_embeddings(""),
        )
        return res["metadatas"][0]
