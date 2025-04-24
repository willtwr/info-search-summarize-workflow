from .base_llm_pipe import BaseLLMPipe
import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Qwen(BaseLLMPipe):
    """Qwen 2.5 language model pipeline implementation.

    This class implements a pipeline for the Qwen 2.5 (3B parameters) instruction-tuned model
    with AWQ quantization for efficient inference. The model is configured to run on CUDA
    with float16 precision.
    
    The pipeline is configured with specific generation parameters:
    - Maximum 4096 new tokens per generation
    - Low temperature (0.1) for focused, deterministic outputs
    - Light repetition penalty (1.05) to prevent redundant text
    - Sampling enabled for natural language generation
    """

    def build_pipe(self) -> None:
        """Configure and build the Qwen pipeline.

        Creates a text generation pipeline with:
        - AWQ quantized model loaded in float16 precision on CUDA
        - Conservative sampling settings for reliable outputs
        - Extended maximum sequence length of 4096 tokens
        - Trust remote code enabled for model-specific optimizations
        """
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct-AWQ", 
            device_map="cuda", 
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct-AWQ")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=4096,
            repetition_penalty=1.05,
            do_sample=True,
            temperature=0.1,
            return_full_text=False
        )
        self.pipe = HuggingFacePipeline(pipeline=pipe)
