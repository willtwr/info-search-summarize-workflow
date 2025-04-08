from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.tools.simple import Tool


def build_my_budget_retriever(retriever: VectorStoreRetriever) -> Tool:
    vector_store_retriever = create_retriever_tool(
        retriever,
        name="malaysia_budget_data_retriever",
        description="Tool for retrieving data regarding Malaysia's budget allocations, spending, subsidies, etc.",
    )
    return vector_store_retriever
