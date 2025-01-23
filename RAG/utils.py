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

from pydantic import BaseModel


logging.set_verbosity(40)
warnings.filterwarnings("ignore", category=FutureWarning)


tag_generation_prompt = """
You are a summary and tags generator,
your role is to look at the document provided,
and generate category tags associated to it,
along with a 5 line summary,
you should limit the number of tags to 3.
"""


class TagsModel(BaseModel):
    tags: list[str]
    summary: str


class RAGtoolkit:
    def __init__(self, collection_name="generic"):
        settings = Settings(is_persistent=True)
        self.splitter = TextSplitter(overlap=True, capacity=1000, trim=True)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter(
            "allenai/specter2", source="hf", load_as="proximity", set_active=True
        )
        self.client = chromadb.Client(settings=settings)
        self._collection = self.client.get_or_create_collection(name=collection_name)
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
        return self.splitter.chunks(doc)

    def generate_meta(self, docs_dir: str) -> list[str]:
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
                        {"role": "system", "content": tag_generation_prompt},
                        {"role": "user", "content": f.read()},
                    ],
                    format=TagsModel.model_json_schema(),
                    options={"num_ctx": 8192},
                )
                print(res["message"]["content"])
                page_level_meta = TagsModel.model_validate_json(
                    res["message"]["content"]
                )
                print(page_level_meta)
                for tag in page_level_meta.tags:
                    tags.add(tag)
                summarys.append(page_level_meta.summary)

        return {"tags": list(tags), "summarys": summarys}

    def add_docs(self, docs: list[str]):
        self._collection.upsert(
            documents=docs,
            ids=["id1", "id2"],
            embeddings=self._generate_embeddings(docs),
        )

    def query_docs(self, query: str) -> QueryResult:
        question_embedding = self._generate_embeddings(query)[0]
        res = self._collection.query(
            query_embeddings=[question_embedding], query_texts=[query]
        )
        return res


kit = RAGtoolkit()

print(kit.generate_meta("outputs/pages/chapter_0"))
