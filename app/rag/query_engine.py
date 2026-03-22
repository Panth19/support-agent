from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine

from app.rag.llm_settings import configure_llama_index
from app.rag.qdrant_store import get_vector_store


def get_query_engine() -> BaseQueryEngine:
    """Create a query engine connected to the existing Qdrant collection.

    Returns:
        A LlamaIndex query engine configured with ``similarity_top_k=5``
        and ``response_mode="compact"``.
    """
    configure_llama_index()

    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    return index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )
