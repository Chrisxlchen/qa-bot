import pytest
from unittest.mock import Mock, MagicMock
from src.retriever import DocumentRetriever
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


class TestDocumentRetriever:
    """Test cases for the DocumentRetriever class."""
    
    def test_retriever_initialization(self):
        """Test DocumentRetriever initialization."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        assert retriever.vector_store == mock_vector_store
        assert retriever.embedding_generator == mock_embedding_generator
    
    def test_retrieve_documents_success(self):
        """Test successful document retrieval."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        # Mock embedding generation
        query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.generate_embeddings.return_value = [query_embedding]
        
        # Mock vector store search results
        search_results = [
            {
                'content': 'Document content 1',
                'metadata': {'source': 'doc1.txt'},
                'distance': 0.1
            },
            {
                'content': 'Document content 2',
                'metadata': {'source': 'doc2.txt'},
                'distance': 0.3
            }
        ]
        mock_vector_store.search.return_value = search_results
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        query = "What is this about?"
        results = retriever.retrieve_documents(query, n_results=2)
        
        assert results == search_results
        mock_embedding_generator.generate_embeddings.assert_called_once_with([query])
        mock_vector_store.search.assert_called_once_with(
            query_embedding=query_embedding,
            n_results=2
        )
    
    def test_retrieve_documents_default_n_results(self):
        """Test document retrieval with default n_results."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        mock_embedding_generator.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_store.search.return_value = []
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        retriever.retrieve_documents("test query")
        
        mock_vector_store.search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            n_results=5  # Default value
        )
    
    def test_retrieve_documents_empty_results(self):
        """Test document retrieval when no results are found."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        mock_embedding_generator.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_vector_store.search.return_value = []
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        results = retriever.retrieve_documents("test query")
        
        assert results == []
    
    def test_format_context_single_document(self):
        """Test formatting context for a single document."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        retrieved_docs = [
            {
                'content': 'This is document content.',
                'metadata': {'source': 'test.txt'},
                'distance': 0.2
            }
        ]
        
        context = retriever.format_context(retrieved_docs)
        
        expected_context = "Document 1 (source: test.txt, relevance: 0.800):\nThis is document content.\n"
        assert context == expected_context
    
    def test_format_context_multiple_documents(self):
        """Test formatting context for multiple documents."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        retrieved_docs = [
            {
                'content': 'First document content.',
                'metadata': {'source': 'doc1.txt'},
                'distance': 0.1
            },
            {
                'content': 'Second document content.',
                'metadata': {'source': 'doc2.txt'},
                'distance': 0.3
            }
        ]
        
        context = retriever.format_context(retrieved_docs)
        
        lines = context.split('\n')
        assert "Document 1 (source: doc1.txt, relevance: 0.900):" in lines[0]
        assert "First document content." in lines[1]
        assert "Document 2 (source: doc2.txt, relevance: 0.700):" in lines[3]
        assert "Second document content." in lines[4]
    
    def test_format_context_empty_list(self):
        """Test formatting context for empty document list."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        context = retriever.format_context([])
        
        assert context == ""
    
    def test_format_context_missing_source(self):
        """Test formatting context when source is missing."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        retrieved_docs = [
            {
                'content': 'Document without source.',
                'metadata': {},  # No source in metadata
                'distance': 0.15
            }
        ]
        
        context = retriever.format_context(retrieved_docs)
        
        assert "Unknown source" in context
        assert "relevance: 0.850" in context
    
    def test_relevance_score_calculation(self):
        """Test relevance score calculation (1 - distance)."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        # Test various distance values
        test_cases = [
            (0.0, 1.000),  # Perfect match
            (0.5, 0.500),  # Medium relevance
            (1.0, 0.000),  # No relevance
        ]
        
        for distance, expected_relevance in test_cases:
            retrieved_docs = [
                {
                    'content': 'Test content',
                    'metadata': {'source': 'test.txt'},
                    'distance': distance
                }
            ]
            
            context = retriever.format_context(retrieved_docs)
            assert f"relevance: {expected_relevance:.3f}" in context
    
    def test_retrieve_documents_embedding_generation_error(self):
        """Test handling of embedding generation errors."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        # Mock embedding generation to raise an exception
        mock_embedding_generator.generate_embeddings.side_effect = Exception("Embedding error")
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        with pytest.raises(Exception, match="Embedding error"):
            retriever.retrieve_documents("test query")
    
    def test_retrieve_documents_vector_search_error(self):
        """Test handling of vector search errors."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        mock_embedding_generator.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        # Mock vector store search to raise an exception
        mock_vector_store.search.side_effect = Exception("Search error")
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        with pytest.raises(Exception, match="Search error"):
            retriever.retrieve_documents("test query")
    
    def test_context_formatting_special_characters(self):
        """Test context formatting with special characters."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embedding_generator = Mock(spec=EmbeddingGenerator)
        
        retriever = DocumentRetriever(mock_vector_store, mock_embedding_generator)
        
        retrieved_docs = [
            {
                'content': 'Content with émojis 🤖 and newlines\nSecond line',
                'metadata': {'source': 'special_chars.txt'},
                'distance': 0.1
            }
        ]
        
        context = retriever.format_context(retrieved_docs)
        
        assert 'émojis 🤖' in context
        assert 'Second line' in context
        assert 'special_chars.txt' in context