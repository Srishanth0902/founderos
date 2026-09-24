from langchain.tools import tool
from rag.vectorstore import load_vector_store
from config import VECTOR_STORE_DIR
import os


def get_retriever():
    """
    Returns the FAISS retriever.
    """
    vectorstore = load_vector_store()
    if not vectorstore:
        raise ValueError("Vector store not initialized. Please build it first.")

    return vectorstore.as_retriever(search_kwargs={"k": 5})


@tool
def founder_knowledge_retriever(query: str) -> str:
    """Searches and returns information about the founder's resume, portfolio, skills, achievements, personality, goals, and projects."""
    try:
        retriever = get_retriever()
    except (FileNotFoundError, ValueError, RuntimeError):
        return "Founder knowledge search is unavailable because the vector store has not been initialized."

    docs = retriever.invoke(query)
    if not docs:
        return "No relevant knowledge found."
    return "\n\n".join(doc.page_content for doc in docs)


def get_retriever_tool():
    """
    Returns a LangChain tool for the retriever.
    """
    if not os.path.exists(VECTOR_STORE_DIR):
        raise ValueError("Vector store not initialized. Run init_rag.py first.")
    return founder_knowledge_retriever
