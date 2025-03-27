from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.tools.simple import Tool


def build_vector_store_retriever(retriever: VectorStoreRetriever) -> Tool:
    vector_store_retriever = create_retriever_tool(
        retriever,
        name="vector_store_retriever",
        description="Information from documents",
    )
    return vector_store_retriever
