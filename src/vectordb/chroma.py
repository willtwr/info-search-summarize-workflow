from uuid import uuid4
from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.text_embedding.stella import Stella
import pymupdf


class ChromaVectorStore:
    """Chroma Vector Database"""
    def __init__(self, embedding_function=Stella()):
        self.embedding_function = embedding_function
        self._build_docs_splitter()
        self.build_vector_store()

    def build_vector_store(self) -> None:
        # Create vector store. Change collection name for different tables
        self.vectorstore = Chroma(
            collection_name="documents",
            collection_metadata={"type": "pdf"},
            embedding_function=self.embedding_function,
            persist_directory="./chromadb"
        )

    def _build_docs_splitter(self, chunk_size: int = 512, chunk_overlap: int = 128) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _docs_splitter(self, docs: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(docs)
    
    def read_pdf(self, path: str) -> List[Document]:
        """Read PDF file and return list of pages"""
        doc = pymupdf.open(path)
        content = []
        for i, page in enumerate(doc):
            #TODO: Add last 128 tokens/words from previous page to the beginning of current page.
            content.append(
                Document(
                    page_content=page.get_text().encode("utf-8"),
                    metadata={
                        "source": path,
                        "page": i
                    }
                )
            )
        
        return content

    def add_documents(self, docs: List[Document]) -> None:
        doc_splits = self._docs_splitter(docs)
        uuids = [str(uuid4()) for _ in range(len(doc_splits))]
        self.vectorstore.add_documents(doc_splits, ids=uuids)

    def get_retriever(self) -> VectorStoreRetriever:
        return self.vectorstore.as_retriever()
