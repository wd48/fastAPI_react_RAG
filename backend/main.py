import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA

# .env 파일에서 환경 변수 로드
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

app = FastAPI()

# React Frontend와 통신을 위한 CORS 설정 (이전과 동일)
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

# RAG 모델 로드 (Hugging Face 모델로 변경)
# 1. 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-nli",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# 2. FAISS 인덱스 로드
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# 3. LLM 모델 변경 (HuggingFaceEndpoint 사용)
# repo_id: Hugging Face에 있는 모델 ID
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it",
    temperature=0.1,
    max_new_tokens=512,
)

# RetrievalQA 체인 생성 (이전과 동일)
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