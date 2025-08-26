import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 환경변수 로드
load_dotenv()

# 1. 문서 로드
loader = TextLoader("text_data.txt", encoding="utf-8")
documents = loader.load()

# 2. 텍스트 분할
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. 임베딩 모델 변경 및 생성
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-nli",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

db = FAISS.from_documents(docs, embeddings)

# 4. FAISS 인덱스 저장
db.save_local("faiss_index")
print("FAISS index created and saved to 'faiss_index' directory successfully.")