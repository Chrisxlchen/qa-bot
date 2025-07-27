import os
from typing import List, Dict
import openai


class LLMHandler:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        
        # Determine the API configuration based on model name
        if model_name.startswith("gpt-"):
            # OpenAI model
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai.OpenAI(api_key=api_key)
        elif model_name.startswith("deepseek"):
            # Deepseek model
            api_key = os.getenv("LLM_API_KEY")
            base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            # Default to OpenAI for backward compatibility
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = openai.OpenAI(api_key=api_key)
        
        self.system_prompt = """You are a helpful assistant that answers questions based on the provided context documents. 
        
Rules:
1. Only answer based on the information provided in the context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite specific documents when possible
4. Be concise but thorough
5. If multiple documents provide relevant information, synthesize them appropriately
"""
    
    def generate_answer(self, query: str, context: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def generate_answer_with_sources(self, query: str, retrieved_docs: List[Dict]) -> Dict[str, str]:
        context = ""
        sources = set()
        
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('source', 'Unknown source')
            sources.add(source)
            content = doc['content']
            distance = doc['distance']
            
            context += f"Document {i} (from {source}):\n{content}\n\n"
        
        answer = self.generate_answer(query, context)
        
        return {
            "answer": answer,
            "sources": list(sources),
            "context_used": len(retrieved_docs)
        }