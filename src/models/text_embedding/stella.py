from typing import List
from sentence_transformers import SentenceTransformer


class Stella:
    """Stella embedding model"""
    def __init__(self):
        self.query_prompt_name = "s2p_query"
        self.build_model()

    def build_model(self):
        self.model = SentenceTransformer(
            "dunzhang/stella_en_1.5B_v5",
            trust_remote_code=True
        ).cuda()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, prompt_name=self.query_prompt_name).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
