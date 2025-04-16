from .smollm2 import SmolLM2
from .qwen import Qwen


# define models here
models = {
    "smollm2": SmolLM2,
    "qwen": Qwen
}


def llm_pipe_factory(model_name="qwen"):
    """Factory function for creating language model pipeline instances.

    This function implements the factory pattern to instantiate different language
    model pipelines based on the requested model name. Currently supports:
    - SmolLM2: A 1.7B parameter instruction-tuned model
    - Qwen: A 3B parameter instruction-tuned model with AWQ quantization

    Args:
        model_name (str, optional): Name of the model to instantiate. Defaults to "qwen".

    Returns:
        BaseChatModel or HuggingFacePipeline: The initialized language model pipeline

    Raises:
        KeyError: If the requested model name is not found in the supported models
    """
    return models[model_name]().get_pipe()
