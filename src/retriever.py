import os
from typing import List, Dict
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore


class DocumentRetriever:
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    def retrieve_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results
        )
        
        return results
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('source', 'Unknown source')
            content = doc['content']
            distance = doc['distance']
            
            context_parts.append(
                f"Document {i} (source: {source}, relevance: {1-distance:.3f}):\n{content}\n"
            )
        
        return "\n".join(context_parts)