import ollama
import asyncio
from utils import RAGtoolkit

client = ollama.AsyncClient()


async def RAGPipeline(query: str) -> str:
    # get the chapter associated to query
    generic_kit = RAGtoolkit()
    res = generic_kit.query_metadata(query)
