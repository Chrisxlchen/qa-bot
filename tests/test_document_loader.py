import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from src.document_loader import DocumentLoader


class TestDocumentLoader:
    """Test cases for the DocumentLoader class."""
    
    def test_document_loader_initialization(self, temp_dir):
        """Test DocumentLoader initialization."""
        loader = DocumentLoader(temp_dir)
        
        assert str(loader.documents_path) == temp_dir
        assert loader.text_splitter.chunk_size == 1000
        assert loader.text_splitter.chunk_overlap == 200
    
    def test_document_loader_default_path(self):
        """Test DocumentLoader initialization with default path."""
        loader = DocumentLoader()
        
        assert str(loader.documents_path) == "./documents"
    
    def test_load_text_file(self, temp_dir):
        """Test loading a text file."""
        loader = DocumentLoader(temp_dir)
        test_file = Path(temp_dir) / "test.txt"
        test_content = "This is a test file content."
        test_file.write_text(test_content)
        
        result = loader.load_text_file(test_file)
        
        assert result == test_content
    
    def test_load_markdown_file(self, temp_dir):
        """Test loading a markdown file."""
        loader = DocumentLoader(temp_dir)
        test_file = Path(temp_dir) / "test.md"
        test_content = "# Test Markdown\n\nThis is markdown content."
        test_file.write_text(test_content)
        
        result = loader.load_markdown(test_file)
        
        assert result == test_content
    
    @patch('pypdf.PdfReader')
    def test_load_pdf_file(self, mock_pdf_reader, temp_dir):
        """Test loading a PDF file."""
        loader = DocumentLoader(temp_dir)
        test_file = Path(temp_dir) / "test.pdf"
        test_file.touch()  # Create empty file
        
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF content from page"
        mock_pdf_reader.return_value.pages = [mock_page, mock_page]
        
        with patch('builtins.open', mock_open()):
            result = loader.load_pdf(test_file)
        
        assert result == "PDF content from pagePDF content from page"
        mock_pdf_reader.assert_called_once()
    
    def test_load_documents_mixed_types(self, sample_documents_dir):
        """Test loading documents of different types."""
        loader = DocumentLoader(sample_documents_dir)
        
        # Add a PDF file for testing
        pdf_file = Path(sample_documents_dir) / "test.pdf"
        pdf_file.touch()
        
        with patch.object(loader, 'load_pdf', return_value="PDF content"):
            documents = loader.load_documents()
        
        assert len(documents) >= 2  # At least text and markdown files
        
        # Check that documents have required fields
        for doc in documents:
            assert 'content' in doc
            assert 'metadata' in doc
            assert 'source' in doc['metadata']
    
    def test_load_documents_empty_directory(self, temp_dir):
        """Test loading documents from empty directory."""
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()
        loader = DocumentLoader(str(empty_dir))
        
        documents = loader.load_documents()
        
        assert documents == []
    
    def test_load_documents_nonexistent_directory(self):
        """Test loading documents from non-existent directory."""
        loader = DocumentLoader("/nonexistent/path")
        
        documents = loader.load_documents()
        
        assert documents == []
    
    @patch('src.document_loader.RecursiveCharacterTextSplitter')
    def test_text_splitting_called(self, mock_splitter, sample_documents_dir):
        """Test that text splitting is called during document loading."""
        loader = DocumentLoader(sample_documents_dir)
        mock_splitter.return_value.split_text.return_value = ["chunk1", "chunk2"]
        
        documents = loader.load_documents()
        
        # Verify that text splitting was called
        assert mock_splitter.return_value.split_text.called
    
    def test_load_documents_with_unsupported_file_type(self, temp_dir):
        """Test loading documents with unsupported file types."""
        loader = DocumentLoader(temp_dir)
        
        # Create an unsupported file type
        unsupported_file = Path(temp_dir) / "test.xyz"
        unsupported_file.write_text("This should be ignored")
        
        documents = loader.load_documents()
        
        # Should not include unsupported file
        assert all('.xyz' not in doc['metadata']['source'] for doc in documents)
    
    def test_document_metadata_structure(self, sample_documents_dir):
        """Test that document metadata has correct structure."""
        loader = DocumentLoader(sample_documents_dir)
        
        documents = loader.load_documents()
        
        for doc in documents:
            assert isinstance(doc['metadata'], dict)
            assert 'source' in doc['metadata']
            assert 'chunk_index' in doc['metadata']
            assert isinstance(doc['metadata']['chunk_index'], int)
    
    def test_load_file_with_encoding_issues(self, temp_dir):
        """Test loading files with potential encoding issues."""
        loader = DocumentLoader(temp_dir)
        test_file = Path(temp_dir) / "encoding_test.txt"
        
        # Write file with UTF-8 content
        test_file.write_text("Test with émojis 🤖 and special chars", encoding='utf-8')
        
        result = loader.load_text_file(test_file)
        
        assert "émojis 🤖" in result
    
    @patch('src.document_loader.DocumentLoader.load_pdf')
    def test_pdf_loading_error_handling(self, mock_load_pdf, temp_dir):
        """Test error handling when PDF loading fails."""
        loader = DocumentLoader(temp_dir)
        pdf_file = Path(temp_dir) / "corrupted.pdf"
        pdf_file.touch()
        
        # Mock PDF loading to raise an exception
        mock_load_pdf.side_effect = Exception("PDF parsing error")
        
        documents = loader.load_documents()
        
        # Should handle the error gracefully and not include the problematic PDF
        pdf_docs = [doc for doc in documents if 'corrupted.pdf' in doc['metadata']['source']]
        assert len(pdf_docs) == 0