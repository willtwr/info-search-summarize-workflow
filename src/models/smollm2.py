from .base_llm import BaseLLM

import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


class SmolLM2(BaseLLM):
    """SmolLM 2 model"""
    def __init__(self):
        self.build_pipe()

    def build_pipe(self) -> None:
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
        self.smollm2pipe = HuggingFacePipeline(pipeline=pipe)

    def get_pipe(self):
        return self.smollm2pipe
