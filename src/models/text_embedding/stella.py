from typing import List
from sentence_transformers import SentenceTransformer


class Stella:
    """Stella EN 1.5B v5 text embedding model.

    This class provides an interface to the Stella EN 1.5B v5 model for generating
    text embeddings. It supports both single-text and batch embedding generation,
    with specialized handling for query-specific embeddings.

    The model runs on CUDA for efficient inference and uses a specific query prompt
    template for optimizing search-related embeddings.

    Attributes:
        query_prompt_name (str): Name of the prompt template for queries
        model (SentenceTransformer): The underlying Stella transformer model
    """

    def __init__(self):
        """Initialize the Stella embedding model.

        Sets up the query prompt configuration and loads the model onto CUDA.
        """
        self.query_prompt_name = "s2p_query"
        self.build_model()

    def build_model(self) -> None:
        """Load and configure the Stella model.

        Initializes the SentenceTransformer with the Stella EN 1.5B v5 weights
        and moves it to CUDA for GPU acceleration.
        """
        self.model = SentenceTransformer(
            "dunzhang/stella_en_1.5B_v5",
            trust_remote_code=True
        ).cuda()

    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a search query.

        Uses a specialized query prompt template to optimize the embedding
        for search purposes.

        Args:
            text: The query text to embed

        Returns:
            A list of floating-point values representing the text embedding
        """
        return self.model.encode(text, prompt_name=self.query_prompt_name).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of documents.

        Processes multiple texts in parallel for efficient embedding generation.

        Args:
            texts: List of document texts to embed

        Returns:
            A list of embeddings, where each embedding is a list of floats
        """
        return self.model.encode(texts).tolist()
