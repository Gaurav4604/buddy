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


logging.set_verbosity(40)
warnings.filterwarnings("ignore", category=FutureWarning)


meta_generation_prompt = """
You are a summary and tags generator,
your role is to look at the document provided,
and generate category tags associated to it,
along with a 5 line summary,
you should limit the number of tags to 3.
"""

summary_merge_prompt = """
You are a summary merge manager,
your role is to look at multi-page summaries,
and merge them into a single summary, and associate
5 tags with the same.
"""


class TagsModel(BaseModel):
    tags: list[str]
    summary: str


class RAGtoolkit:
    def __init__(self, chapter_num: int = 0):
        settings = Settings(is_persistent=True)
        self.splitter = TextSplitter(overlap=True, capacity=500, trim=True)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
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
                    format=TagsModel.model_json_schema(),
                    options={"num_ctx": 8192},
                )
                page_level_meta = TagsModel.model_validate_json(
                    res["message"]["content"]
                )
                for tag in page_level_meta.tags:
                    tags.add(tag)
                summarys.append(page_level_meta.summary)

        res = self.ollama_client.chat(
            model="deepseek-r1",
            messages=[
                {"role": "system", "content": summary_merge_prompt},
                {"role": "user", "content": summarys[0] + ", ".join(summarys[1:])},
            ],
            format=TagsModel.model_json_schema(),
            options={"num_ctx": 8192},
        )
        overall_summary = TagsModel.model_validate_json(res["message"]["content"])

        return {"tags": list(tags), "summarys": summarys, "summary": overall_summary}

    # chunks for the given chapter to be added
    def add_docs(self, docs: list[str]):
        self._doc_collection.upsert(
            documents=docs,
            ids=[uuid.uuid1() for _ in range(docs)],
            embeddings=self._generate_embeddings(docs),
        )

    def query_docs(self, query: str) -> QueryResult:
        question_embedding = self._generate_embeddings(query)[0]
        res = self._collection.query(
            query_embeddings=[question_embedding], query_texts=[query]
        )
        return res


# each chapter will have its own kit access
kit = RAGtoolkit()

print(kit.generate_meta("outputs/pages/chapter_0"))

# data = open("outputs/pages/chapter_0/page_4.txt", "r", encoding="utf-8").read()

# chunks = kit.generate_chunks(data)

# for chunk in chunks:
#     print(chunk)
#     print("-----NEW CHUNK------")


"""
summaries obtained
tags generated
chunks extracted

next steps:
- merge with main
- embed docs, pack with page no. info for each embedding metadata (to be done on chunk)
- add fn to add meta-data associated to file (concate tags with summary -> dump into metadata collection db with chapter no. as metadata)
- add fn to query meta-data associated with file

- final pipeline for QnA RAG?
    - get chapters associated to query (using metadata collection)
    - query required chapter collections with query again
    - use results to ask LLM question
    - respond with answer
"""
