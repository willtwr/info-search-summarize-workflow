from .base_llm_pipe import BaseLLMPipe
import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


class SmolLM2(BaseLLMPipe):
    """SmolLM 2 language model pipeline implementation.

    This class implements a pipeline for the SmolLM 2 (1.7B parameters) instruction-tuned model.
    The model is configured to run on CUDA with bfloat16 precision for efficient inference.
    
    The pipeline is configured with specific generation parameters:
    - Maximum 1024 new tokens per generation
    - Temperature of 0.55 for controlled randomness
    - Sampling enabled for more diverse outputs
    """

    def build_pipe(self) -> None:
        """Configure and build the SmolLM 2 pipeline.

        Creates a text generation pipeline with:
        - Model loaded in bfloat16 precision on CUDA
        - Sampling-based generation with temperature control
        - Maximum sequence length of 1024 tokens
        """
        pipe = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            torch_dtype=torch.bfloat16,
            device="cuda",
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.55,
            return_full_text=False
        )
        self.pipe = HuggingFacePipeline(pipeline=pipe)
