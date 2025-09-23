import os
import sys

# backend 디렉토리를 Python 경로에 추가하여 app 모듈을 찾을 수 있도록 합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.services.document_service import load_documents_from_pdf, split_documents
from backend.app.services.vector_store_service import get_embeddings
from langchain_community.vectorstores import Qdrant
from backend.app.core import config

def main():
    """
    PDF 문서들을 로드하고, 청크로 분할한 뒤, 벡터로 변환하여 Qdrant DB에 저장합니다.
    """
    print("PDF 문서 로드를 시작합니다...")
    documents = load_documents_from_pdf()
    if not documents:
        print("로드할 PDF 문서가 없습니다. pdf_data 디렉토리를 확인해주세요.")
        return

    print(f"총 {len(documents)}개의 문서를 로드했습니다.")

    print("문서 분할을 시작합니다...")
    docs = split_documents(documents)
    print(f"총 {len(docs)}개의 청크로 분할되었습니다.")

    print("임베딩 모델을 초기화합니다...")
    embeddings = get_embeddings()

    print("Qdrant 데이터베이스에 데이터를 저장합니다...")
    Qdrant.from_documents(
        docs,
        embeddings,
        url=config.QDRANT_URL,
        collection_name=config.COLLECTION_NAME,
        force_recreate=True,  # 실행 시마다 컬렉션을 새로 만듭니다.
    )

    print(f"'{config.COLLECTION_NAME}' 컬렉션에 데이터 저장을 완료했습니다.")

if __name__ == "__main__":
    main()
