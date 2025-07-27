# 🤖 QA Bot - Document Question & Answer System

A local Q&A bot that can answer questions about your documents using LLM and vector search technology.

## ✨ Features

- **Document Support**: PDF, Markdown, and TXT files
- **Vector Search**: ChromaDB for efficient similarity search
- **Multiple Embedding Options**: OpenAI or Hugging Face embeddings
- **LLM Integration**: OpenAI GPT models for natural language answers
- **REST API**: FastAPI backend with endpoints for indexing and querying
- **Web Interface**: Streamlit frontend for easy interaction
- **Source Attribution**: Shows which documents were used to answer questions

## 🏗️ Architecture

```
Documents → Document Loader → Embeddings → Vector Store → Retriever → LLM → Answer
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or create the project directory
cd qa-bot

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Add Documents

Place your documents in the `documents/` folder:
```
documents/
├── research_paper.pdf
├── notes.md
└── documentation.txt
```

### 3. Run the Application

**Option A: FastAPI Backend + Streamlit Frontend**
```bash
# Terminal 1: Start FastAPI server
python app.py

# Terminal 2: Start Streamlit frontend
streamlit run streamlit_app.py
```

**Option B: Python Script (CLI)**
```python
from src.qa_bot import QABot

# Initialize bot
bot = QABot()

# Index documents
bot.index_documents()

# Ask questions
result = bot.ask("What are the main topics discussed?")
print(result["answer"])
```

## 📁 Project Structure

```
qa-bot/
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # Load and chunk documents
│   ├── embeddings.py         # Generate embeddings
│   ├── vector_store.py       # ChromaDB integration
│   ├── retriever.py          # Document retrieval
│   ├── llm.py               # LLM integration
│   └── qa_bot.py            # Main QA bot class
├── documents/               # Place your documents here
├── chroma_db/              # Vector database storage
├── app.py                  # FastAPI server
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt        # Dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## 🔧 Configuration

Edit `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-3.5-turbo
CHROMA_PERSIST_DIRECTORY=./chroma_db
DOCUMENTS_PATH=./documents
```

## 📡 API Endpoints

- `POST /index` - Index documents in the vector database
- `POST /ask` - Ask a question and get an answer
- `GET /stats` - Get system statistics
- `GET /` - Health check

### Example API Usage

```bash
# Index documents
curl -X POST "http://localhost:8000/index" \
     -H "Content-Type: application/json" \
     -d '{"force_reindex": false}'

# Ask a question
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic?", "n_results": 5}'
```

## 🛠️ Customization

### Use Hugging Face Embeddings Instead of OpenAI

```python
bot = QABot(
    embedding_model_type="huggingface",
    embedding_model_name="all-MiniLM-L6-v2"
)
```

### Different LLM Models

```python
bot = QABot(llm_model="gpt-4")
```

## 📝 Usage Tips

1. **Document Quality**: Better formatted documents lead to better answers
2. **Chunk Size**: The system splits documents into 1000-character chunks with 200-character overlap
3. **Query Style**: Ask specific questions for better results
4. **Relevance**: The system shows relevance scores for retrieved documents

## 🔍 Troubleshooting

- **"No documents found"**: Check that documents are in the `documents/` folder
- **"API connection error"**: Ensure the FastAPI server is running on port 8000
- **"OpenAI API error"**: Verify your API key in the `.env` file
- **"ChromaDB error"**: Delete the `chroma_db/` folder and re-index

## 🚀 Deployment Options

### Local Development
- FastAPI + Streamlit (current setup)

### Cloud Deployment
- Docker containerization
- Deploy FastAPI to platforms like Railway, Render, or AWS
- Host Streamlit frontend separately or use FastAPI static files

## 📚 Dependencies

- **FastAPI**: Web framework for the API
- **Streamlit**: Frontend interface
- **LangChain**: Document processing and text splitting
- **ChromaDB**: Vector database
- **OpenAI**: Embeddings and LLM
- **pypdf**: PDF processing
- **sentence-transformers**: Alternative embeddings

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.