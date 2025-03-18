from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract class for LLM model"""
    @abstractmethod
    def build_pipe(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_pipe(self):
        raise NotImplementedError
