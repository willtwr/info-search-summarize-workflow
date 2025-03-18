from .base_llm import BaseLLM
import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Qwen(BaseLLM):
    """Qwen2.5-3B-Instruct-AWQ model"""
    def __init__(self):
        self.build_pipe()

    def build_pipe(self) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct-AWQ", 
            device_map="cuda", 
            torch_dtype=torch.float16,
            trust_remote_code=True
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

    def get_pipe(self):
        return self.pipe
