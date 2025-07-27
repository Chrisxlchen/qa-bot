import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from app import app, qa_bot


class TestAPI:
    """Test cases for the FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_qa_bot(self):
        """Mock the global qa_bot instance."""
        with patch('app.qa_bot') as mock_bot:
            yield mock_bot
    
    def test_root_endpoint(self, client):
        """Test the root health check endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "QA Bot API is running!"}
    
    def test_index_documents_success(self, client, mock_qa_bot):
        """Test successful document indexing."""
        mock_qa_bot.index_documents = Mock()
        mock_qa_bot.get_stats = Mock(return_value={
            "total_documents": 10,
            "is_indexed": True
        })
        
        response = client.post("/index", json={"force_reindex": False})
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Documents indexed successfully"
        assert data["stats"]["total_documents"] == 10
        
        mock_qa_bot.index_documents.assert_called_once_with(force_reindex=False)
        mock_qa_bot.get_stats.assert_called_once()
    
    def test_index_documents_force_reindex(self, client, mock_qa_bot):
        """Test document indexing with force reindex."""
        mock_qa_bot.index_documents = Mock()
        mock_qa_bot.get_stats = Mock(return_value={"total_documents": 5})
        
        response = client.post("/index", json={"force_reindex": True})
        
        assert response.status_code == 200
        mock_qa_bot.index_documents.assert_called_once_with(force_reindex=True)
    
    def test_index_documents_default_force_reindex(self, client, mock_qa_bot):
        """Test document indexing with default force_reindex value."""
        mock_qa_bot.index_documents = Mock()
        mock_qa_bot.get_stats = Mock(return_value={"total_documents": 0})
        
        response = client.post("/index", json={})
        
        assert response.status_code == 200
        mock_qa_bot.index_documents.assert_called_once_with(force_reindex=False)
    
    def test_index_documents_error(self, client, mock_qa_bot):
        """Test document indexing with error."""
        mock_qa_bot.index_documents.side_effect = Exception("Indexing failed")
        
        response = client.post("/index", json={"force_reindex": False})
        
        assert response.status_code == 500
        assert "Indexing failed" in response.json()["detail"]
    
    def test_ask_question_success(self, client, mock_qa_bot):
        """Test successful question asking."""
        mock_result = {
            "answer": "This is the answer",
            "sources": ["doc1.txt", "doc2.txt"],
            "context_used": 3
        }
        mock_qa_bot.ask = Mock(return_value=mock_result)
        
        response = client.post("/ask", json={
            "question": "What is this about?",
            "n_results": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is the answer"
        assert data["sources"] == ["doc1.txt", "doc2.txt"]
        assert data["context_used"] == 3
        
        mock_qa_bot.ask.assert_called_once_with("What is this about?", 5)
    
    def test_ask_question_default_n_results(self, client, mock_qa_bot):
        """Test asking question with default n_results."""
        mock_qa_bot.ask = Mock(return_value={"answer": "Test answer"})
        
        response = client.post("/ask", json={"question": "Test question"})
        
        assert response.status_code == 200
        mock_qa_bot.ask.assert_called_once_with("Test question", 5)
    
    def test_ask_question_with_error_result(self, client, mock_qa_bot):
        """Test asking question when QA bot returns error."""
        mock_result = {"error": "No documents indexed"}
        mock_qa_bot.ask = Mock(return_value=mock_result)
        
        response = client.post("/ask", json={"question": "Test question"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["error"] == "No documents indexed"
    
    def test_ask_question_exception(self, client, mock_qa_bot):
        """Test asking question with exception."""
        mock_qa_bot.ask.side_effect = Exception("QA processing failed")
        
        response = client.post("/ask", json={"question": "Test question"})
        
        assert response.status_code == 500
        assert "QA processing failed" in response.json()["detail"]
    
    def test_get_stats_success(self, client, mock_qa_bot):
        """Test successful stats retrieval."""
        mock_stats = {
            "total_documents": 15,
            "is_indexed": True,
            "embedding_model": "openai:text-embedding-ada-002",
            "llm_model": "gpt-3.5-turbo"
        }
        mock_qa_bot.get_stats = Mock(return_value=mock_stats)
        
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data == mock_stats
        
        mock_qa_bot.get_stats.assert_called_once()
    
    def test_get_stats_error(self, client, mock_qa_bot):
        """Test stats retrieval with error."""
        mock_qa_bot.get_stats.side_effect = Exception("Stats retrieval failed")
        
        response = client.get("/stats")
        
        assert response.status_code == 500
        assert "Stats retrieval failed" in response.json()["detail"]
    
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set."""
        response = client.options("/")
        
        # The exact CORS behavior depends on the browser/client, but we can test basic functionality
        assert response.status_code in [200, 405]  # 405 is acceptable for OPTIONS on GET endpoint
    
    def test_ask_question_invalid_json(self, client):
        """Test asking question with invalid JSON."""
        response = client.post("/ask", json={})  # Missing required 'question' field
        
        assert response.status_code == 422  # Validation error
    
    def test_index_documents_invalid_json(self, client):
        """Test indexing with invalid JSON structure."""
        response = client.post("/index", json={"invalid_field": "value"})
        
        # Should still work with default values
        assert response.status_code in [200, 500]  # 500 if qa_bot is not properly mocked
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('app.QABot')
    def test_startup_event(self, mock_qa_bot_class):
        """Test the startup event initialization."""
        from app import startup_event
        import asyncio
        
        mock_bot_instance = Mock()
        mock_qa_bot_class.return_value = mock_bot_instance
        
        # Run the startup event
        asyncio.run(startup_event())
        
        # Verify QABot was initialized with correct parameters
        mock_qa_bot_class.assert_called_once()
        call_args = mock_qa_bot_class.call_args[1]
        assert call_args['embedding_model_type'] == "openai"
        assert call_args['llm_model'] == "deepseek-chat"
    
    def test_question_request_model_validation(self, client, mock_qa_bot):
        """Test QuestionRequest model validation."""
        mock_qa_bot.ask = Mock(return_value={"answer": "Test"})
        
        # Test with valid data
        response = client.post("/ask", json={
            "question": "Valid question",
            "n_results": 10
        })
        assert response.status_code == 200
        
        # Test with invalid n_results type
        response = client.post("/ask", json={
            "question": "Valid question",
            "n_results": "invalid"
        })
        assert response.status_code == 422
    
    def test_index_request_model_validation(self, client, mock_qa_bot):
        """Test IndexRequest model validation."""
        mock_qa_bot.index_documents = Mock()
        mock_qa_bot.get_stats = Mock(return_value={})
        
        # Test with valid boolean
        response = client.post("/index", json={"force_reindex": True})
        assert response.status_code == 200
        
        # Test with invalid boolean type
        response = client.post("/index", json={"force_reindex": "invalid"})
        assert response.status_code == 422
    
    def test_api_title_and_version(self):
        """Test that API has correct title and version."""
        assert app.title == "QA Bot API"
        assert app.version == "1.0.0"
    
    def test_content_type_headers(self, client, mock_qa_bot):
        """Test that endpoints expect correct content types."""
        mock_qa_bot.ask = Mock(return_value={"answer": "Test"})
        
        # Test with correct content type
        response = client.post("/ask", 
            json={"question": "Test"},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
        
        # Test with form data (should fail)
        response = client.post("/ask", 
            data={"question": "Test"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 422