from transformers import AutoTokenizer
from adapters import AutoAdapterModel

import chromadb
from chromadb.config import Settings
from chromadb.api.types import QueryResult
from transformers.utils import logging
import warnings

logging.set_verbosity(40)
warnings.filterwarnings("ignore", category=FutureWarning)


class RAGtoolkit:
    def __init__(self, collection_name="generic"):
        settings = Settings(is_persistent=True)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter(
            "allenai/specter2", source="hf", load_as="proximity", set_active=True
        )
        self.client = chromadb.Client(settings=settings)
        self._collection = self.client.get_or_create_collection(name=collection_name)

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
