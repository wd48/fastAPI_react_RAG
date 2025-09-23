from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.rag_service import qa_chain

# API 라우터 생성
router = APIRouter()

# 요청 모델 정의
class QueryRequest(BaseModel):
    query: str

# 챗봇과의 대화를 처리하는 엔드포인트
@router.post("/chat")
async def chat_with_bot(request: QueryRequest):
    """사용자 질문에 대한 답변을 생성하는 API 엔드포인트"""
    try:
        response = qa_chain.invoke({"query": request.query})
        return {"response": response['result']}
    except Exception as e:
        # 실제 운영 환경에서는 더 상세한 로깅이 필요합니다.
        print(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
