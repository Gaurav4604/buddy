import chromadb
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    get_response_synthesizer,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


ollama_embedding = OllamaEmbedding(
    model_name="llama3.1",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)


Settings.embed_model = ollama_embedding
Settings.llm = Ollama(model="llama3.1", request_timeout=360)

db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.get_or_create_collection("data")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

vector_index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)


query_engine = vector_index.as_query_engine()
response_synthesizer = get_response_synthesizer(
    Ollama(model="llama3.1", request_timeout=360)
)


retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=1,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever, response_synthesizer=response_synthesizer
)

response = query_engine.query(
    "generate 5 difficult questions from the corpus",
)
print(response)
