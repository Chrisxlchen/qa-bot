import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.llm import LLMHandler


class TestLLMHandler:
    """Test cases for the LLMHandler class."""
    
    def test_llm_handler_initialization(self):
        """Test LLMHandler initialization."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI') as mock_openai:
            
            handler = LLMHandler("gpt-4")
            
            assert handler.model_name == "gpt-4"
            mock_openai.assert_called_once_with(api_key='test-key')
    
    def test_llm_handler_default_model(self):
        """Test LLMHandler initialization with default model."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI'):
            
            handler = LLMHandler()
            
            assert handler.model_name == "gpt-3.5-turbo"
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI'):
            
            handler = LLMHandler()
            
            assert "helpful assistant" in handler.system_prompt
            assert "context documents" in handler.system_prompt
            assert "Rules:" in handler.system_prompt
    
    @patch('openai.OpenAI')
    def test_generate_answer_success(self, mock_openai):
        """Test successful answer generation."""
        # Setup mock
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated answer"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            handler = LLMHandler("gpt-3.5-turbo")
            
            query = "What is this about?"
            context = "This is context information."
            
            answer = handler.generate_answer(query, context)
            
            assert answer == "Generated answer"
            
            # Verify API call
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args[1]
            assert call_args['model'] == "gpt-3.5-turbo"
            assert call_args['temperature'] == 0.3
            assert call_args['max_tokens'] == 500
            assert len(call_args['messages']) == 2
            assert call_args['messages'][0]['role'] == 'system'
            assert call_args['messages'][1]['role'] == 'user'
    
    @patch('openai.OpenAI')
    def test_generate_answer_api_error(self, mock_openai):
        """Test answer generation with API error."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            handler = LLMHandler()
            
            answer = handler.generate_answer("query", "context")
            
            assert "Error generating answer: API Error" in answer
    
    @patch('openai.OpenAI')
    def test_generate_answer_with_sources_success(self, mock_openai):
        """Test successful answer generation with sources."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Answer with sources"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            handler = LLMHandler()
            
            query = "What is this about?"
            retrieved_docs = [
                {
                    'content': 'First document content',
                    'metadata': {'source': 'doc1.txt'},
                    'distance': 0.1
                },
                {
                    'content': 'Second document content',
                    'metadata': {'source': 'doc2.txt'},
                    'distance': 0.2
                },
                {
                    'content': 'Third document content',
                    'metadata': {'source': 'doc1.txt'},  # Duplicate source
                    'distance': 0.3
                }
            ]
            
            result = handler.generate_answer_with_sources(query, retrieved_docs)
            
            assert result['answer'] == "Answer with sources"
            assert set(result['sources']) == {'doc1.txt', 'doc2.txt'}  # Should deduplicate
            assert result['context_used'] == 3
    
    def test_generate_answer_with_sources_empty_docs(self):
        """Test answer generation with empty document list."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI') as mock_openai:
            
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="No context answer"))]
            mock_client.chat.completions.create.return_value = mock_response
            
            handler = LLMHandler()
            
            result = handler.generate_answer_with_sources("query", [])
            
            assert result['answer'] == "No context answer"
            assert result['sources'] == []
            assert result['context_used'] == 0
    
    def test_generate_answer_with_sources_missing_metadata(self):
        """Test answer generation when documents have missing metadata."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI') as mock_openai:
            
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Answer"))]
            mock_client.chat.completions.create.return_value = mock_response
            
            handler = LLMHandler()
            
            retrieved_docs = [
                {
                    'content': 'Content without source',
                    'metadata': {},  # No source
                    'distance': 0.1
                }
            ]
            
            result = handler.generate_answer_with_sources("query", retrieved_docs)
            
            assert result['answer'] == "Answer"
            assert 'Unknown source' in result['sources']
            assert result['context_used'] == 1
    
    @patch('openai.OpenAI')
    def test_context_formatting_in_prompt(self, mock_openai):
        """Test that context is properly formatted in the prompt."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Answer"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            handler = LLMHandler()
            
            retrieved_docs = [
                {
                    'content': 'First content',
                    'metadata': {'source': 'file1.txt'},
                    'distance': 0.1
                },
                {
                    'content': 'Second content',
                    'metadata': {'source': 'file2.txt'},
                    'distance': 0.2
                }
            ]
            
            handler.generate_answer_with_sources("test query", retrieved_docs)
            
            # Check the user message content
            call_args = mock_client.chat.completions.create.call_args[1]
            user_message = call_args['messages'][1]['content']
            
            assert "Document 1 (from file1.txt):" in user_message
            assert "First content" in user_message
            assert "Document 2 (from file2.txt):" in user_message
            assert "Second content" in user_message
            assert "Question: test query" in user_message
    
    def test_answer_content_stripping(self):
        """Test that answer content is properly stripped of whitespace."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI') as mock_openai:
            
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="  Answer with spaces  \n"))]
            mock_client.chat.completions.create.return_value = mock_response
            
            handler = LLMHandler()
            
            answer = handler.generate_answer("query", "context")
            
            assert answer == "Answer with spaces"
    
    @patch('openai.OpenAI')
    def test_model_parameter_usage(self, mock_openai):
        """Test that the correct model parameter is used in API calls."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Answer"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            handler = LLMHandler("gpt-4-turbo")
            
            handler.generate_answer("query", "context")
            
            call_args = mock_client.chat.completions.create.call_args[1]
            assert call_args['model'] == "gpt-4-turbo"
    
    def test_temperature_and_max_tokens_parameters(self):
        """Test that temperature and max_tokens are set correctly."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI') as mock_openai:
            
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Answer"))]
            mock_client.chat.completions.create.return_value = mock_response
            
            handler = LLMHandler()
            
            handler.generate_answer("query", "context")
            
            call_args = mock_client.chat.completions.create.call_args[1]
            assert call_args['temperature'] == 0.3
            assert call_args['max_tokens'] == 500
    
    def test_sources_deduplication(self):
        """Test that duplicate sources are properly deduplicated."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}), \
             patch('openai.OpenAI') as mock_openai:
            
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Answer"))]
            mock_client.chat.completions.create.return_value = mock_response
            
            handler = LLMHandler()
            
            retrieved_docs = [
                {'content': 'Content 1', 'metadata': {'source': 'same_file.txt'}, 'distance': 0.1},
                {'content': 'Content 2', 'metadata': {'source': 'same_file.txt'}, 'distance': 0.2},
                {'content': 'Content 3', 'metadata': {'source': 'different_file.txt'}, 'distance': 0.3}
            ]
            
            result = handler.generate_answer_with_sources("query", retrieved_docs)
            
            assert len(result['sources']) == 2
            assert 'same_file.txt' in result['sources']
            assert 'different_file.txt' in result['sources']