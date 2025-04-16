from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.tools.simple import Tool


def build_my_budget_retriever(retriever: VectorStoreRetriever) -> Tool:
    """Create a tool for retrieving information from the vector store.

    This function creates a LangChain tool that wraps a vector store retriever,
    specifically configured for searching budget-related information. It provides
    a natural language interface to the vector database.

    Args:
        retriever: The base vector store retriever to wrap as a tool

    Returns:
        Tool: A LangChain tool configured for budget data retrieval
    """
    vector_store_retriever = create_retriever_tool(
        retriever,
        name="malaysia_budget_data_retriever",
        description="Tool for retrieving data regarding Malaysia's budget allocations, spending, subsidies, etc.",
    )
    return vector_store_retriever
