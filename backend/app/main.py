from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api import chat

# FastAPI 애플리케이션 생성 및 설정
def create_app() -> FastAPI:
    """FastAPI 애플리케이션을 생성하고 설정합니다."""
    app = FastAPI(
        title="RAG 챗봇 API",
        description="Ollama와 Qdrant를 이용한 문서 기반 챗봇",
        version="1.0.0"
    )

    # CORS 설정
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite 기본 포트
        "http://127.0.0.1:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API 라우터 포함
    app.include_router(chat.router, prefix="/api")

    # 기본 상태 확인 엔드포인트
    @app.get("/", tags=["Root"])
    async def read_root():
        """서버 상태를 확인하는 기본 엔드포인트"""
        return {"message": "RAG API 서버가 정상적으로 실행 중입니다."}

    return app

# FastAPI 애플리케이션 인스턴스 생성
app = create_app()
