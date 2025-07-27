import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch
from src.qa_bot import QABot
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documents_dir(temp_dir):
    """Create sample documents for testing."""
    docs_dir = Path(temp_dir) / "documents"
    docs_dir.mkdir()
    
    # Create sample text file
    (docs_dir / "sample.txt").write_text("This is a sample text document for testing.")
    
    # Create sample markdown file
    (docs_dir / "sample.md").write_text("# Sample Markdown\n\nThis is a sample markdown document.")
    
    return str(docs_dir)


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings API."""
    with patch('openai.OpenAI') as mock_client:
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.return_value.embeddings.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_openai_llm():
    """Mock OpenAI LLM API."""
    with patch('openai.OpenAI') as mock_client:
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test answer"))]
        mock_client.return_value.chat.completions.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def qa_bot_instance(temp_dir, mock_openai_embeddings, mock_openai_llm):
    """Create a QABot instance for testing."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        bot = QABot(
            documents_path=str(Path(temp_dir) / "documents"),
            persist_directory=str(Path(temp_dir) / "chroma_db"),
            embedding_model_type="openai",
            embedding_model_name="text-embedding-ada-002",
            llm_model="gpt-3.5-turbo"
        )
        yield bot