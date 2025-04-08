import io
from PIL import Image

from typing import Optional
from models.llm.llm_pipe_factory import llm_pipe_factory
from tools.newssearch import news_search
from tools.websearch import web_search
from tools.vector_store_retriever import build_my_budget_retriever
from tools.tools_cond import tools_condition
from agents.websearcher.websercher import WebSearcherAgent
from agents.summarizer.summarizer import SummarizerAgent

from langchain_core.language_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.vectorstores import VectorStoreRetriever


class WorkflowGraph:
    """Workflow Graph"""
    def __init__(
            self,
            model_name: str = "qwen",
            model: Optional[BaseChatModel] = None,
            vectorstore: Optional[VectorStoreRetriever] = None
    ):
        # Load model
        self.model_name = model_name
        if model is None:
            self.build_model()
        else:
            self.model = model

        # Build vectorstore retriever
        self.vectorstore_retriever = build_my_budget_retriever(vectorstore) if vectorstore else None
        
        # Build workflow graph
        self.build_graph()

    def build_model(self) -> None:
        """Build the LLM model"""
        llm = llm_pipe_factory(self.model_name)
        if not isinstance(llm, BaseChatModel):
            self.model = ChatHuggingFace(llm=llm)
        else:
            self.model = llm

    def build_graph(self) -> None:
        """Build the graph of the agent"""
        memory = MemorySaver()
        graph_builder = StateGraph(MessagesState)

        # Agents
        websearcher_agent = WebSearcherAgent(model=self.model)
        summarizer_agent = SummarizerAgent(model=self.model)

        # tools
        tools = [news_search, web_search]
        if self.vectorstore_retriever:
            tools.append(self.vectorstore_retriever)

        tool_node = ToolNode(tools=tools)
        websearcher_agent.bind_tools(tools)

        graph_builder.add_node("websearcher", websearcher_agent)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_node("summarizer", summarizer_agent)
        
        graph_builder.set_entry_point("websearcher")
        graph_builder.add_conditional_edges("websearcher", tools_condition)
        graph_builder.add_edge("tools", "summarizer")
        graph_builder.add_edge("summarizer", END)

        self.graph = graph_builder.compile(checkpointer=memory)
        print(self.graph.get_graph().draw_mermaid())
        image = Image.open(io.BytesIO(self.graph.get_graph().draw_mermaid_png()))
        image.save("./assets/workflow-graph.png")

    def __call__(self):
        return self.graph
