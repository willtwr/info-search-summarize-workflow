from uuid import uuid4
from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.text_embedding.stella import Stella


class ChromaVectorStore:
    """Chroma Vector Database integration for document storage and retrieval.

    This class provides an interface to the Chroma vector database for storing and retrieving
    documents with vector embeddings. It handles document splitting, PDF reading, and
    vector store management.

    The store uses a default embedding model (Stella) but can be configured with other
    embedding functions. Documents are split into chunks for more effective retrieval.

    Attributes:
        embedding_function: The function used to generate embeddings for documents
        text_splitter: Splitter for breaking documents into manageable chunks
        vectorstore: The underlying Chroma vector store instance
    """

    def __init__(self, embedding_function=Stella()):
        """Initialize the vector store with an embedding function.

        Args:
            embedding_function: Function to generate embeddings. Defaults to Stella model.
        """
        self.embedding_function = embedding_function
        self._build_docs_splitter()
        self.build_vector_store()

    def build_vector_store(self) -> None:
        """Initialize the Chroma vector store.

        Creates a new Chroma collection for documents with the configured embedding function
        and persistence settings.
        """
        self.vectorstore = Chroma(
            collection_name="documents",
            collection_metadata={"type": "pdf"},
            embedding_function=self.embedding_function,
            persist_directory="./chromadb"
        )

    def _build_docs_splitter(self, chunk_size: int = 512, chunk_overlap: int = 128) -> None:
        """Configure the document splitter with specified parameters.

        Args:
            chunk_size: Maximum size of text chunks in tokens. Defaults to 512.
            chunk_overlap: Number of overlapping tokens between chunks. Defaults to 128.
        """
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _docs_splitter(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks for storage.

        Args:
            docs: List of documents to split

        Returns:
            List of split document chunks
        """
        return self.text_splitter.split_documents(docs)
    
    def add_documents(self, docs: List[Document]) -> None:
        """Add documents to the vector store.

        Documents are split into chunks before being added to the store. Each chunk
        gets a unique UUID.

        Args:
            docs: List of documents to add to the store
        """
        doc_splits = self._docs_splitter(docs)
        uuids = [str(uuid4()) for _ in range(len(doc_splits))]
        self.vectorstore.add_documents(doc_splits, ids=uuids)

    def get_retriever(self) -> VectorStoreRetriever:
        """Get a retriever interface to the vector store.

        Returns:
            A VectorStoreRetriever instance for querying the store
        """
        return self.vectorstore.as_retriever()
