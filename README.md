## 시스템 아키텍처 (System Architecture)
- Qdrant Vector DB
  - 문서(Documents)를 임베딩(Embeddings)으로 변환하여 저장합니다.
  - 사용자의 질문과 가장 유사한 문서를 빠르게 찾아내는 역할을 합니다.
- FastAPI Backend
  - 사용자의 질문을 받아서 처리하는 API 서버
  - Qdrant에서 관련 문서를 검색하고, 이 문서들을 **Ollama LLM (Large Language Model)**에 전달하여 답변을 생성합니다.
- React Frontend
  - 사용자가 텍스트를 입력하고 챗봇의 답변을 볼 수 있는 UI(User Interface)
  - API 요청을 FastAPI 백엔드로 보냅니다.
