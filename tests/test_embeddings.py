import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from src.embeddings import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test cases for the EmbeddingGenerator class."""
    
    def test_embedding_generator_openai_initialization(self):
        """Test EmbeddingGenerator initialization with OpenAI."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI') as mock_openai:
            
            generator = EmbeddingGenerator("openai", "text-embedding-ada-002")
            
            assert generator.model_type == "openai"
            assert generator.model_name == "text-embedding-ada-002"
            mock_openai.assert_called_once_with(api_key='test-key')
    
    def test_embedding_generator_huggingface_initialization(self):
        """Test EmbeddingGenerator initialization with Hugging Face."""
        with patch('src.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator("huggingface", "all-MiniLM-L6-v2")
            
            assert generator.model_type == "huggingface"
            assert generator.model_name == "all-MiniLM-L6-v2"
            mock_st.assert_called_once_with("all-MiniLM-L6-v2")
    
    def test_embedding_generator_huggingface_default_model(self):
        """Test EmbeddingGenerator with default Hugging Face model."""
        with patch('src.embeddings.SentenceTransformer') as mock_st:
            generator = EmbeddingGenerator("huggingface", None)
            
            mock_st.assert_called_once_with("all-MiniLM-L6-v2")
    
    @patch('openai.OpenAI')
    def test_generate_openai_embeddings_success(self, mock_openai):
        """Test successful OpenAI embeddings generation."""
        # Setup mock
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536)
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            generator = EmbeddingGenerator("openai", "text-embedding-ada-002")
            
            texts = ["text1", "text2"]
            embeddings = generator.generate_embeddings(texts)
            
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 1536
            assert embeddings[0] == [0.1] * 1536
            assert embeddings[1] == [0.2] * 1536
            
            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-ada-002",
                input=texts
            )
    
    @patch('openai.OpenAI')
    def test_generate_openai_embeddings_batch_processing(self, mock_openai):
        """Test OpenAI embeddings generation with batch processing."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Create a side effect that returns the right number of embeddings based on input
        def mock_create_embeddings(**kwargs):
            input_texts = kwargs['input']
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in input_texts]
            return mock_response
        
        mock_client.embeddings.create.side_effect = mock_create_embeddings
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            generator = EmbeddingGenerator("openai", "text-embedding-ada-002")
            
            # Create 150 texts to test batching (batch_size = 100)
            texts = [f"text{i}" for i in range(150)]
            embeddings = generator.generate_embeddings(texts)
            
            assert len(embeddings) == 150
            # Should make 2 API calls (100 + 50)
            assert mock_client.embeddings.create.call_count == 2
    
    @patch('openai.OpenAI')
    def test_generate_openai_embeddings_api_error(self, mock_openai):
        """Test OpenAI embeddings generation with API error."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("API Error")
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            generator = EmbeddingGenerator("openai", "text-embedding-ada-002")
            
            texts = ["text1", "text2"]
            embeddings = generator.generate_embeddings(texts)
            
            # Should return zero embeddings on error
            assert len(embeddings) == 2
            assert all(emb == [0.0] * 1536 for emb in embeddings)
    
    def test_generate_huggingface_embeddings_success(self):
        """Test successful Hugging Face embeddings generation."""
        with patch('src.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator("huggingface", "all-MiniLM-L6-v2")
            
            texts = ["text1", "text2"]
            embeddings = generator.generate_embeddings(texts)
            
            assert len(embeddings) == 2
            assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)
    
    def test_generate_huggingface_embeddings_error(self):
        """Test Hugging Face embeddings generation with error."""
        with patch('src.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.side_effect = Exception("Model Error")
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator("huggingface", "all-MiniLM-L6-v2")
            
            texts = ["text1", "text2"]
            embeddings = generator.generate_embeddings(texts)
            
            # Should return zero embeddings on error
            assert len(embeddings) == 2
            assert all(emb == [0.0] * 384 for emb in embeddings)
    
    def test_generate_embeddings_unsupported_model_type(self):
        """Test generating embeddings with unsupported model type."""
        generator = EmbeddingGenerator("unsupported", "model")
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            generator.generate_embeddings(["text"])
    
    def test_get_embedding_dimension_openai(self):
        """Test getting embedding dimension for OpenAI model."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI'):
            
            generator = EmbeddingGenerator("openai", "text-embedding-ada-002")
            
            assert generator.get_embedding_dimension() == 1536
    
    def test_get_embedding_dimension_huggingface(self):
        """Test getting embedding dimension for Hugging Face model."""
        with patch('src.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator("huggingface", "bert-base-uncased")
            
            assert generator.get_embedding_dimension() == 768
            mock_model.get_sentence_embedding_dimension.assert_called_once()
    
    def test_get_embedding_dimension_default_fallback(self):
        """Test getting embedding dimension with default fallback."""
        generator = EmbeddingGenerator("unknown", "model")
        
        assert generator.get_embedding_dimension() == 384
    
    def test_empty_text_list(self):
        """Test generating embeddings for empty text list."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI') as mock_openai:
            
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.data = []
            mock_client.embeddings.create.return_value = mock_response
            
            generator = EmbeddingGenerator("openai", "text-embedding-ada-002")
            
            embeddings = generator.generate_embeddings([])
            
            assert embeddings == []
    
    def test_single_text_embedding(self):
        """Test generating embedding for single text."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI') as mock_openai:
            
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.5] * 1536)]
            mock_client.embeddings.create.return_value = mock_response
            
            generator = EmbeddingGenerator("openai", "text-embedding-ada-002")
            
            embeddings = generator.generate_embeddings(["single text"])
            
            assert len(embeddings) == 1
            assert embeddings[0] == [0.5] * 1536