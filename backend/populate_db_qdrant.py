import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant

# 환경변수 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# 1. PDF 문서 로드
pdf_data_path = os.path.join(os.path.dirname(__file__), "pdf_data")
documents = []
for file in os.listdir(pdf_data_path):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_data_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

# 2. 텍스트 분할
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Ollama 임베딩 모델 생성
embeddings = OllamaEmbeddings(
    model="gemma:2b",
    base_url=os.getenv("OLLAMA_BASE_URL")
)

# 4. Qdrant에 문서 저장
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=os.getenv("QDRANT_URL"),
    collection_name="rag_collection",
)

print("Database populated with new data in collection 'rag_collection' successfully.")
