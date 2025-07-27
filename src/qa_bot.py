import os
from typing import Dict, List
from .document_loader import DocumentLoader
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .retriever import DocumentRetriever
from .llm import LLMHandler


class QABot:
    def __init__(self, 
                 documents_path: str = "./documents",
                 persist_directory: str = "./chroma_db",
                 embedding_model_type: str = "openai",
                 embedding_model_name: str = None,
                 llm_model: str = "gpt-3.5-turbo"):
        
        self.document_loader = DocumentLoader(documents_path)
        self.embedding_generator = EmbeddingGenerator(embedding_model_type, embedding_model_name)
        self.vector_store = VectorStore(persist_directory)
        self.retriever = DocumentRetriever(self.vector_store, self.embedding_generator)
        self.llm_handler = LLMHandler(llm_model)
        
        self.is_indexed = False
    
    def index_documents(self, force_reindex: bool = False):
        if self.is_indexed and not force_reindex:
            print("Documents already indexed. Use force_reindex=True to reindex.")
            return
        
        if force_reindex:
            self.vector_store.reset_collection()
        
        print("Loading documents...")
        documents = self.document_loader.load_documents()
        
        if not documents:
            print("No documents found to index.")
            return
        
        print(f"Generating embeddings for {len(documents)} document chunks...")
        texts = [doc['content'] for doc in documents]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        print("Storing documents in vector database...")
        self.vector_store.add_documents(documents, embeddings)
        
        self.is_indexed = True
        print(f"Indexing complete! {len(documents)} chunks indexed.")
    
    def ask(self, question: str, n_results: int = 5) -> Dict:
        if not self.is_indexed and self.vector_store.get_collection_count() == 0:
            return {
                "error": "No documents indexed. Please run index_documents() first."
            }
        
        print(f"Retrieving relevant documents for: {question}")
        retrieved_docs = self.retriever.retrieve_documents(question, n_results)
        
        if not retrieved_docs:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": [],
                "context_used": 0
            }
        
        print("Generating answer...")
        result = self.llm_handler.generate_answer_with_sources(question, retrieved_docs)
        
        return result
    
    def get_stats(self) -> Dict:
        return {
            "total_documents": self.vector_store.get_collection_count(),
            "is_indexed": self.is_indexed,
            "embedding_model": f"{self.embedding_generator.model_type}:{self.embedding_generator.model_name}",
            "llm_model": self.llm_handler.model_name
        }