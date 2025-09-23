import os
from dotenv import load_dotenv

# backend 디렉토리의 .env 파일을 로드합니다.
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# 환경 변수 또는 기본값을 설정합니다.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3n:latest")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")

# 데이터 경로 설정
# 이 파일(config.py)의 위치를 기준으로 backend/pdf_data 경로를 계산합니다.
PDF_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'pdf_data'))
