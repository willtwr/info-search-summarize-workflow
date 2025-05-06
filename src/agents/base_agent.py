from abc import ABC, abstractmethod
from typing import Optional
from models.llm.llm_pipe_factory import llm_pipe_factory
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langgraph.graph import MessagesState


class BaseAgent(ABC):
    """Abstract Base Agent Class for implementing AI agents in the workflow.

    This class serves as the foundation for all agents in the system. It handles model
    initialization and system prompt loading while defining the interface that all
    agents must implement.

    Attributes:
        model_name (str): Name of the language model to use
        model (BaseChatModel): The underlying language model instance
        sysprompt_path (str): Path to the system prompt file
        thinking_mode (bool): Flag to indicate if use thinking mode

    Args:
        model_name (str, optional): Name of the model to use. Defaults to "qwen".
        model (BaseChatModel, optional): Pre-initialized model instance. Defaults to None.
        sysprompt_path (str, optional): Path to system prompt file. Defaults to None.
    """
    def __init__(
            self,
            model_name: str = "qwen",
            model: Optional[BaseChatModel] = None,
            sysprompt_path: Optional[str] = None,
            thinking_mode: bool = False
    ):
        # Load system prompt
        self.sysprompt_path = sysprompt_path
        self.sys_prompt = None
        self.load_system_prompt()

        if not thinking_mode:
            if isinstance(self.sys_prompt, str):
                self.sys_prompt = self.sys_prompt + "\n\n/no_think"
            elif isinstance(self.sys_prompt, PromptTemplate):
                self.sys_prompt.template = self.sys_prompt.template + "\n\n/no_think"
            else:
                raise ValueError("Either invalid system prompt type or system prompt not initialized. Must be str or PromptTemplate.")

        # Model initialization
        if model is None:
            self.model_name = model_name
            self.build_model()
        else:
            self.model_name = None
            self.model = model

    @abstractmethod
    def load_system_prompt(self) -> None:
        """Load and initialize the system prompt for the agent.
        
        This method should be implemented by subclasses to load their specific
        system prompts from files or other sources.
        """
        pass

    def build_model(self) -> None:
        """Initialize the language model for the agent.
        
        Creates a new language model instance using the factory pattern.
        If the model is not a ChatModel, wraps it in ChatHuggingFace.
        """
        llm = llm_pipe_factory(self.model_name)
        if isinstance(llm, BaseChatModel):
            self.model = llm
        else:
            self.model = ChatHuggingFace(llm=llm)

    @abstractmethod
    def invoke(self, state: MessagesState) -> dict:
        """Process the current message state and generate a response.

        Args:
            state (MessagesState): Current state containing message history and context

        Returns:
            dict: Response containing new messages or actions to be taken
        """
        pass
    
    def __call__(self, state: MessagesState) -> dict:
        """Make the agent callable, delegating to invoke method.
        
        Args:
            state (MessagesState): Current state containing message history and context
            
        Returns:
            dict: Response containing new messages or actions to be taken
        """
        return self.invoke(state)
