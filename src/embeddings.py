import os
from typing import List
import openai
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
    def __init__(self, model_type: str = "openai", model_name: str = None):
        self.model_type = model_type
        
        if model_type == "openai":
            self.model_name = model_name or "text-embedding-ada-002"
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif model_type == "deepseek":
            self.model_name = model_name or "deepseek-embedding"
            api_key = os.getenv("LLM_API_KEY")
            base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        elif model_type == "huggingface":
            self.model_name = model_name or "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.model_type == "openai":
            return self._generate_openai_embeddings(texts)
        elif self.model_type == "deepseek":
            return self._generate_deepseek_embeddings(texts)
        elif self.model_type == "huggingface":
            return self._generate_huggingface_embeddings(texts)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                embeddings.extend([[0.0] * 1536] * len(batch))
        
        return embeddings
    
    def _generate_deepseek_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                embeddings.extend([[0.0] * 1536] * len(batch))
        
        return embeddings
    
    def _generate_huggingface_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero embeddings with default dimension for HuggingFace models
            try:
                dimension = self.model.get_sentence_embedding_dimension()
            except:
                dimension = 384  # Default fallback
            return [[0.0] * dimension] * len(texts)
    
    def get_embedding_dimension(self) -> int:
        if self.model_type == "openai":
            return 1536
        elif self.model_type == "deepseek":
            return 1536
        elif self.model_type == "huggingface":
            return self.model.get_sentence_embedding_dimension()
        return 384