from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama
from .vector_store_service import get_vector_store
from ..core import config

def get_llm() -> Ollama:
    """Ollama LLM을 초기화하고 반환합니다."""
    return Ollama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )

def create_qa_chain() -> RetrievalQA:
    """RetrievalQA 체인을 생성하고 반환합니다."""
    llm = get_llm()
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

# 애플리케이션 시작 시 QA 체인을 미리 생성합니다.
qa_chain = create_qa_chain()
