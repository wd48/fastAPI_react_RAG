import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient

# .env 파일에서 환경 변수 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = FastAPI()

# React Frontend와 통신을 위한 CORS 설정
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# RAG 모델 로드 (Ollama와 Qdrant로 변경)
# 1. Ollama 임베딩 모델 로드
embeddings = OllamaEmbeddings(
    model="bge-m3:latest",
    base_url=os.getenv("OLLAMA_BASE_URL")
)

# 2. Qdrant 클라이언트 및 vector_store 인스턴스 생성
client = QdrantClient(url=os.getenv("QDRANT_URL"))
vector_store = Qdrant(
    client=client,
    collection_name="rag_collection",
    embeddings=embeddings,
)
retriever = vector_store.as_retriever()

# 3. Ollama LLM 모델 변경
llm = Ollama(
    model="gemma3n:latest",
    base_url=os.getenv("OLLAMA_BASE_URL")
)

# RetrievalQA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)

@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    try:
        response = qa_chain.invoke({"query": request.query})
        return {"response": response['result']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)