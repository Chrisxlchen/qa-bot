import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import uuid


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection_name = "qa_documents"
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Dict[str, str]], embeddings: List[List[float]]):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            doc_id = doc.get('chunk_id', str(uuid.uuid4()))
            ids.append(doc_id)
            texts.append(doc['content'])
            
            # Handle both flat and nested metadata structures
            if 'metadata' in doc:
                # Nested structure (from tests)
                source = doc['metadata'].get('source', doc.get('source', 'Unknown'))
                chunk_id = doc['metadata'].get('chunk_id', doc.get('chunk_id', doc_id))
            else:
                # Flat structure (from actual usage)
                source = doc.get('source', 'Unknown')
                chunk_id = doc.get('chunk_id', doc_id)
            
            metadatas.append({
                'source': source,
                'chunk_id': chunk_id
            })
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def get_collection_count(self) -> int:
        return self.collection.count()
    
    def reset_collection(self):
        self.client.delete_collection(self.collection_name)
        self._initialize_collection()
        print("Collection reset successfully")