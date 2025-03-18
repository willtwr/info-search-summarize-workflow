from .smollm2 import SmolLM2
from .qwen import Qwen


# define models here
models = {
    "smollm2": SmolLM2,
    "qwen": Qwen
}


def llm_factory(model_name="qwen"):
    """Factory function for LLM models"""
    return models[model_name]().get_pipe()
