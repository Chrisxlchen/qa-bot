import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from src.vector_store import VectorStore


class TestVectorStore:
    """Test cases for the VectorStore class."""
    
    @patch('chromadb.PersistentClient')
    def test_vector_store_initialization(self, mock_client, temp_dir):
        """Test VectorStore initialization."""
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        
        store = VectorStore(temp_dir)
        
        assert store.persist_directory == temp_dir
        assert store.collection_name == "qa_documents"
        mock_client.assert_called_once()
        mock_client.return_value.get_collection.assert_called_once_with("qa_documents")
    
    @patch('chromadb.PersistentClient')
    def test_vector_store_create_new_collection(self, mock_client, temp_dir):
        """Test VectorStore creating new collection when none exists."""
        mock_client.return_value.get_collection.side_effect = Exception("Collection not found")
        mock_collection = Mock()
        mock_client.return_value.create_collection.return_value = mock_collection
        
        store = VectorStore(temp_dir)
        
        mock_client.return_value.create_collection.assert_called_once_with(
            name="qa_documents",
            metadata={"hnsw:space": "cosine"}
        )
        assert store.collection == mock_collection
    
    @patch('chromadb.PersistentClient')
    def test_add_documents(self, mock_client, temp_dir):
        """Test adding documents to the vector store."""
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        
        store = VectorStore(temp_dir)
        
        documents = [
            {'content': 'Document 1', 'metadata': {'source': 'doc1.txt'}},
            {'content': 'Document 2', 'metadata': {'source': 'doc2.txt'}}
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        store.add_documents(documents, embeddings)
        
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        
        assert len(call_args['ids']) == 2
        assert call_args['embeddings'] == embeddings
        assert call_args['documents'] == ['Document 1', 'Document 2']
        assert len(call_args['metadatas']) == 2
    
    @patch('chromadb.PersistentClient')
    def test_search_documents(self, mock_client, temp_dir):
        """Test searching documents in the vector store."""
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        
        # Mock search results
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'distances': [[0.1, 0.3]],
            'documents': [['Content 1', 'Content 2']],
            'metadatas': [[{'source': 'doc1.txt'}, {'source': 'doc2.txt'}]]
        }
        
        store = VectorStore(temp_dir)
        
        query_embedding = [0.1, 0.2, 0.3]
        results = store.search(query_embedding, n_results=2)
        
        assert len(results) == 2
        assert results[0]['content'] == 'Content 1'
        assert results[0]['metadata']['source'] == 'doc1.txt'
        assert results[0]['distance'] == 0.1
        assert results[1]['content'] == 'Content 2'
        assert results[1]['distance'] == 0.3
        
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=2,
            include=['documents', 'metadatas', 'distances']
        )
    
    @patch('chromadb.PersistentClient')
    def test_search_no_results(self, mock_client, temp_dir):
        """Test searching when no results are found."""
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        
        # Mock empty search results
        mock_collection.query.return_value = {
            'ids': [[]],
            'distances': [[]],
            'documents': [[]],
            'metadatas': [[]]
        }
        
        store = VectorStore(temp_dir)
        
        results = store.search([0.1, 0.2, 0.3], n_results=5)
        
        assert results == []
    
    @patch('chromadb.PersistentClient')
    def test_get_collection_count(self, mock_client, temp_dir):
        """Test getting collection count."""
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_client.return_value.get_collection.return_value = mock_collection
        
        store = VectorStore(temp_dir)
        
        count = store.get_collection_count()
        
        assert count == 42
        mock_collection.count.assert_called_once()
    
    @patch('chromadb.PersistentClient')
    def test_reset_collection(self, mock_client, temp_dir):
        """Test resetting the collection."""
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        mock_client.return_value.create_collection.return_value = mock_collection
        
        store = VectorStore(temp_dir)
        
        # After deletion, get_collection should fail and create_collection should be called
        mock_client.return_value.get_collection.side_effect = Exception("Collection not found")
        
        store.reset_collection()
        
        mock_client.return_value.delete_collection.assert_called_once_with("qa_documents")
        mock_client.return_value.create_collection.assert_called_with(
            name="qa_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    @patch('chromadb.PersistentClient')
    def test_add_documents_empty_list(self, mock_client, temp_dir):
        """Test adding empty list of documents."""
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        
        store = VectorStore(temp_dir)
        
        store.add_documents([], [])
        
        mock_collection.add.assert_called_once_with(
            ids=[],
            embeddings=[],
            documents=[],
            metadatas=[]
        )
    
    @patch('chromadb.PersistentClient')
    def test_add_documents_mismatched_lengths(self, mock_client, temp_dir):
        """Test adding documents with mismatched lengths."""
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        
        store = VectorStore(temp_dir)
        
        documents = [{'content': 'Doc 1', 'metadata': {'source': 'doc1.txt'}}]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Mismatched length
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((IndexError, ValueError)):
            store.add_documents(documents, embeddings)
    
    @patch('chromadb.PersistentClient')
    def test_search_with_default_n_results(self, mock_client, temp_dir):
        """Test searching with default number of results."""
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'distances': [[0.1]],
            'documents': [['Content 1']],
            'metadatas': [[{'source': 'doc1.txt'}]]
        }
        
        store = VectorStore(temp_dir)
        
        # Test default n_results (should be 5)
        store.search([0.1, 0.2, 0.3])
        
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=5,
            include=['documents', 'metadatas', 'distances']
        )
    
    @patch('chromadb.PersistentClient')
    def test_collection_persistence_settings(self, mock_client, temp_dir):
        """Test that collection is created with correct persistence settings."""
        mock_client.return_value.get_collection.side_effect = Exception("Collection not found")
        mock_collection = Mock()
        mock_client.return_value.create_collection.return_value = mock_collection
        
        store = VectorStore(temp_dir)
        
        # Verify ChromaDB client was created with correct settings
        mock_client.assert_called_once()
        init_call = mock_client.call_args
        assert init_call[1]['path'] == temp_dir
        assert init_call[1]['settings'].anonymized_telemetry is False
        assert init_call[1]['settings'].allow_reset is True
    
    @patch('chromadb.PersistentClient')
    def test_document_id_generation(self, mock_client, temp_dir):
        """Test that document IDs are generated correctly."""
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        
        store = VectorStore(temp_dir)
        
        documents = [
            {'content': 'Doc 1', 'metadata': {'source': 'doc1.txt'}},
            {'content': 'Doc 2', 'metadata': {'source': 'doc2.txt'}}
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        store.add_documents(documents, embeddings)
        
        call_args = mock_collection.add.call_args[1]
        ids = call_args['ids']
        
        # Check that IDs are strings and unique
        assert len(ids) == 2
        assert all(isinstance(id_val, str) for id_val in ids)
        assert len(set(ids)) == 2  # All IDs should be unique