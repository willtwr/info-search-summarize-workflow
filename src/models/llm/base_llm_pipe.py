from abc import ABC, abstractmethod


class BaseLLMPipe(ABC):
    """Abstract class for LLM Pipelines"""
    def __init__(self):
        self.build_pipe()

    @abstractmethod
    def build_pipe(self):
        pass
    
    def get_pipe(self):
        return self.pipe
