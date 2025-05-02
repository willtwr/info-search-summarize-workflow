from .smollm2 import SmolLM2
from .qwen import Qwen
from .qwen3 import Qwen3


# define models here
models = {
    "smollm2": SmolLM2,
    "qwen": Qwen,
    "qwen3": Qwen3
}


def llm_pipe_factory(model_name="qwen"):
    """Factory function for creating language model pipeline instances.

    This function implements the factory pattern to instantiate different language
    model pipelines based on the requested model name. Currently supports:
    - SmolLM2: A 1.7B parameter instruction-tuned model
    - Qwen: A 3B parameter instruction-tuned model with AWQ quantization
    - Qwen3: A 1.7/4B parameter instruction-tuned model (some with AWQ quantization)

    Args:
        model_name (str, optional): Name of the model to instantiate. Defaults to "qwen".

    Returns:
        BaseChatModel or HuggingFacePipeline: The initialized language model pipeline

    Raises:
        KeyError: If the requested model name is not found in the supported models
    """
    return models[model_name]().get_pipe()
