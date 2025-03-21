from abc import ABC, abstractmethod
from typing import Optional
from models.llm_factory import llm_factory
from langchain_huggingface import ChatHuggingFace
from langchain_core.language_models import BaseChatModel
from langgraph.graph import MessagesState


class BaseAgent(ABC):
    """Abstract Base Agent Class"""
    def __init__(
            self,
            model_name: str = "qwen",
            model: Optional[BaseChatModel] = None,
            sysprompt_path: Optional[str] = None
    ):
        # Model initialization
        if model is None:
            self.model_name = model_name
            self.build_model()
        else:
            self.model_name = None
            self.model = model

        # Load system prompt
        self.sysprompt_path = sysprompt_path
        self.load_system_prompt()

    @abstractmethod
    def load_system_prompt(self) -> None:
        pass

    def build_model(self) -> None:
        """Build the LLM model"""
        llm = llm_factory(self.model_name)
        if isinstance(llm, BaseChatModel):
            self.model = llm
        else:
            self.model = ChatHuggingFace(llm=llm)

    @abstractmethod
    def invoke(self, state: MessagesState) -> dict:
        pass
    
    def __call__(self, state: MessagesState) -> dict:
        return self.invoke(state)
