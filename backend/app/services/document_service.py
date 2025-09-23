import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from ..core import config

def load_documents_from_pdf(pdf_path: str = config.PDF_DATA_PATH) -> List[Document]:
    """지정된 디렉토리의 모든 PDF 파일에서 문서를 로드합니다."""
    documents = []
    if not os.path.isdir(pdf_path):
        print(f"Error: Directory not found at {pdf_path}")
        return documents
        
    for file in os.listdir(pdf_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(pdf_path, file)
            try:
                loader = PyPDFLoader(full_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """문서를 작은 청크로 분할합니다."""
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)
