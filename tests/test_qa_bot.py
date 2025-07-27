import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.qa_bot import QABot


class TestQABot:
    """Test cases for the main QABot class."""
    
    def test_qa_bot_initialization(self, temp_dir):
        """Test QABot initialization with custom parameters."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            bot = QABot(
                documents_path=temp_dir,
                persist_directory=f"{temp_dir}/chroma",
                embedding_model_type="openai",
                embedding_model_name="text-embedding-ada-002",
                llm_model="gpt-4"
            )
            
            assert bot.document_loader.documents_path.name == os.path.basename(temp_dir)
            assert bot.embedding_generator.model_type == "openai"
            assert bot.embedding_generator.model_name == "text-embedding-ada-002"
            assert bot.llm_handler.model_name == "gpt-4"
            assert not bot.is_indexed
    
    def test_qa_bot_default_initialization(self):
        """Test QABot initialization with default parameters."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            bot = QABot()
            
            assert str(bot.document_loader.documents_path) == "./documents"
            assert bot.vector_store.persist_directory == "./chroma_db"
            assert bot.embedding_generator.model_type == "openai"
            assert bot.embedding_generator.model_name == "text-embedding-ada-002"
            assert bot.llm_handler.model_name == "gpt-3.5-turbo"
    
    def test_index_documents_first_time(self, qa_bot_instance, sample_documents_dir):
        """Test indexing documents for the first time."""
        # Mock dependencies
        qa_bot_instance.document_loader.load_documents = Mock(return_value=[
            {'content': 'Sample content', 'metadata': {'source': 'test.txt'}}
        ])
        qa_bot_instance.embedding_generator.generate_embeddings = Mock(return_value=[[0.1] * 1536])
        qa_bot_instance.vector_store.add_documents = Mock()
        
        qa_bot_instance.index_documents()
        
        assert qa_bot_instance.is_indexed
        qa_bot_instance.document_loader.load_documents.assert_called_once()
        qa_bot_instance.embedding_generator.generate_embeddings.assert_called_once()
        qa_bot_instance.vector_store.add_documents.assert_called_once()
    
    def test_index_documents_already_indexed(self, qa_bot_instance):
        """Test indexing when documents are already indexed."""
        qa_bot_instance.is_indexed = True
        qa_bot_instance.document_loader.load_documents = Mock()
        
        qa_bot_instance.index_documents()
        
        qa_bot_instance.document_loader.load_documents.assert_not_called()
    
    def test_index_documents_force_reindex(self, qa_bot_instance):
        """Test force reindexing documents."""
        qa_bot_instance.is_indexed = True
        qa_bot_instance.vector_store.reset_collection = Mock()
        qa_bot_instance.document_loader.load_documents = Mock(return_value=[
            {'content': 'Sample content', 'metadata': {'source': 'test.txt'}}
        ])
        qa_bot_instance.embedding_generator.generate_embeddings = Mock(return_value=[[0.1] * 1536])
        qa_bot_instance.vector_store.add_documents = Mock()
        
        qa_bot_instance.index_documents(force_reindex=True)
        
        qa_bot_instance.vector_store.reset_collection.assert_called_once()
        qa_bot_instance.document_loader.load_documents.assert_called_once()
    
    def test_index_documents_no_documents_found(self, qa_bot_instance):
        """Test indexing when no documents are found."""
        qa_bot_instance.document_loader.load_documents = Mock(return_value=[])
        qa_bot_instance.embedding_generator.generate_embeddings = Mock()
        
        qa_bot_instance.index_documents()
        
        assert not qa_bot_instance.is_indexed
        qa_bot_instance.embedding_generator.generate_embeddings.assert_not_called()
    
    def test_ask_question_success(self, qa_bot_instance):
        """Test asking a question successfully."""
        qa_bot_instance.is_indexed = True
        qa_bot_instance.vector_store.get_collection_count = Mock(return_value=5)
        qa_bot_instance.retriever.retrieve_documents = Mock(return_value=[
            {'content': 'Sample content', 'metadata': {'source': 'test.txt'}, 'distance': 0.1}
        ])
        qa_bot_instance.llm_handler.generate_answer_with_sources = Mock(return_value={
            'answer': 'Test answer',
            'sources': ['test.txt'],
            'context_used': 1
        })
        
        result = qa_bot_instance.ask("What is this about?")
        
        assert result['answer'] == 'Test answer'
        assert result['sources'] == ['test.txt']
        assert result['context_used'] == 1
    
    def test_ask_question_not_indexed(self, qa_bot_instance):
        """Test asking a question when documents are not indexed."""
        qa_bot_instance.is_indexed = False
        qa_bot_instance.vector_store.get_collection_count = Mock(return_value=0)
        
        result = qa_bot_instance.ask("What is this about?")
        
        assert 'error' in result
        assert 'No documents indexed' in result['error']
    
    def test_ask_question_no_relevant_documents(self, qa_bot_instance):
        """Test asking a question when no relevant documents are found."""
        qa_bot_instance.is_indexed = True
        qa_bot_instance.vector_store.get_collection_count = Mock(return_value=5)
        qa_bot_instance.retriever.retrieve_documents = Mock(return_value=[])
        
        result = qa_bot_instance.ask("What is this about?")
        
        assert 'No relevant documents found' in result['answer']
        assert result['sources'] == []
        assert result['context_used'] == 0
    
    def test_get_stats(self, qa_bot_instance):
        """Test getting bot statistics."""
        qa_bot_instance.vector_store.get_collection_count = Mock(return_value=10)
        qa_bot_instance.is_indexed = True
        
        stats = qa_bot_instance.get_stats()
        
        assert stats['total_documents'] == 10
        assert stats['is_indexed'] is True
        assert stats['embedding_model'] == 'openai:text-embedding-ada-002'
        assert stats['llm_model'] == 'gpt-3.5-turbo'
    
    def test_ask_with_custom_n_results(self, qa_bot_instance):
        """Test asking a question with custom number of results."""
        qa_bot_instance.is_indexed = True
        qa_bot_instance.vector_store.get_collection_count = Mock(return_value=5)
        qa_bot_instance.retriever.retrieve_documents = Mock(return_value=[])
        
        qa_bot_instance.ask("What is this about?", n_results=10)
        
        qa_bot_instance.retriever.retrieve_documents.assert_called_with("What is this about?", 10)