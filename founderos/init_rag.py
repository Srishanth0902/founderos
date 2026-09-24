from rag.loader import load_documents
from rag.splitter import split_documents
from rag.vectorstore import build_vector_store
from config import KNOWLEDGE_DIR

def init_vector_store():
    print("Loading documents...")
    docs = load_documents(KNOWLEDGE_DIR)
    print("Splitting documents...")
    chunks = split_documents(docs)
    print("Building vector store...")
    build_vector_store(chunks)
    print("Vector store initialized successfully.")

if __name__ == "__main__":
    init_vector_store()
