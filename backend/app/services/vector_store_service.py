from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from ..core import config
# 벡터 저장소 및 임베딩 모델 초기화 서비스
def get_embeddings() -> OllamaEmbeddings:
    """Ollama 임베딩 모델을 초기화하고 반환합니다."""
    return OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )

# Qdrant 클라이언트 초기화 함수
def get_qdrant_client() -> QdrantClient:
    """Qdrant 클라이언트를 초기화하고 반환합니다."""
    return QdrantClient(url=config.QDRANT_URL)

# 벡터 저장소 초기화 함수
def get_vector_store() -> Qdrant:
    """Qdrant 벡터 저장소를 초기화하고 반환합니다."""
    client = get_qdrant_client()
    embeddings = get_embeddings()
    return Qdrant(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embeddings=embeddings,
    )
