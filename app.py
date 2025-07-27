import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from src.qa_bot import QABot

load_dotenv()

app = FastAPI(title="QA Bot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_bot = None

class QuestionRequest(BaseModel):
    question: str
    n_results: int = 5

class IndexRequest(BaseModel):
    force_reindex: bool = False

@app.on_event("startup")
async def startup_event():
    global qa_bot
    
    documents_path = os.getenv("DOCUMENTS_PATH", "./documents")
    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    embedding_model_type = os.getenv("EMBEDDING_MODEL_TYPE", "openai")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", None)
    llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    qa_bot = QABot(
        documents_path=documents_path,
        persist_directory=persist_directory,
        embedding_model_type=embedding_model_type,
        embedding_model_name=embedding_model_name,
        llm_model=llm_model
    )
    
    print("QA Bot initialized successfully!")

@app.get("/")
async def root():
    return {"message": "QA Bot API is running!"}

@app.post("/index")
async def index_documents(request: IndexRequest):
    try:
        qa_bot.index_documents(force_reindex=request.force_reindex)
        stats = qa_bot.get_stats()
        return {"message": "Documents indexed successfully", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        result = qa_bot.ask(request.question, request.n_results)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    try:
        stats = qa_bot.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)