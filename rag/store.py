import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext


ollama_embedding = OllamaEmbedding(
    model_name="llama3.1",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

Settings.embed_model = ollama_embedding
Settings.llm = Ollama(model="llama3.1", request_timeout=360)

documents = SimpleDirectoryReader("./data").load_data()

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("data")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

vector_index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
