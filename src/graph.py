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
    """Main workflow orchestrator for information search and summarization.

    This class creates and manages the workflow graph that coordinates the interaction
    between different components (agents, tools, and models) of the system. It uses
    LangGraph for workflow management and supports both web search and vector store
    retrieval capabilities.

    The workflow consists of:
    1. WebSearcher agent for query interpretation and tool selection
    2. Tool execution nodes for information retrieval
    3. Summarizer agent for condensing retrieved information

    Attributes:
        model_name (str): Name of the language model to use
        model (BaseChatModel): The language model instance
        vectorstore_retriever (Tool): Vector store retrieval tool if configured
        graph (StateGraph): The compiled workflow graph
    """

    def __init__(
            self,
            model_name: str = "qwen",
            model: Optional[BaseChatModel] = None,
            vectorstore: Optional[VectorStoreRetriever] = None
    ):
        """Initialize the workflow graph.

        Args:
            model_name: Name of the LLM to use. Defaults to "qwen".
            model: Pre-initialized model instance. Optional.
            vectorstore: Vector store retriever for document search. Optional.
        """
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
        """Initialize the language model.

        Creates a new language model instance using the factory pattern,
        wrapping it in ChatHuggingFace if needed.
        """
        llm = llm_pipe_factory(self.model_name)
        if not isinstance(llm, BaseChatModel):
            self.model = ChatHuggingFace(llm=llm)
        else:
            self.model = llm

    def build_graph(self) -> None:
        """Construct the workflow graph.

        Creates and configures the workflow graph with:
        - Memory-based checkpointing
        - WebSearcher and Summarizer agents
        - Tool nodes for search operations
        - Conditional edges for workflow control
        
        Also generates and saves a visualization of the graph structure.
        """
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
        image.save("./docs/assets/workflow-graph.png")

    def __call__(self):
        """Make the workflow graph callable.
        
        Returns:
            The compiled workflow graph ready for execution
        """
        return self.graph
