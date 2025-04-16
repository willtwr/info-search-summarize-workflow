from abc import ABC, abstractmethod


class BaseLLMPipe(ABC):
    """Abstract base class for Language Model Pipeline implementations.
    
    This class defines the interface for creating language model pipelines that can be
    used with the LangChain framework. It provides a consistent way to initialize and
    access different language models through a pipeline abstraction.

    All LLM implementations in the system should inherit from this class and implement
    the build_pipe method to configure their specific model pipeline.

    Attributes:
        pipe: The underlying language model pipeline instance
    """

    def __init__(self):
        """Initialize the LLM pipeline.
        
        Calls build_pipe() which must be implemented by subclasses to set up
        their specific model pipeline configuration.
        """
        self.build_pipe()

    @abstractmethod
    def build_pipe(self):
        """Build and configure the language model pipeline.
        
        This method must be implemented by subclasses to:
        1. Load their specific model
        2. Configure any model-specific parameters
        3. Create and store the pipeline in self.pipe
        """
        pass
    
    def get_pipe(self):
        """Get the configured pipeline instance.
        
        Returns:
            The initialized language model pipeline
        """
        return self.pipe
