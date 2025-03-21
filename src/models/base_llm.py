from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract class for LLM model"""
    def __init__(self):
        self.build_pipe()

    @abstractmethod
    def build_pipe(self):
        pass
    
    @abstractmethod
    def get_pipe(self):
        pass
