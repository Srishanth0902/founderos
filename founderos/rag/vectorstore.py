import os
from langchain_community.vectorstores import FAISS
from rag.embeddings import get_embeddings_model
from config import VECTOR_STORE_DIR

def build_vector_store(chunks):
    """
    Builds and saves the FAISS vector store from document chunks.
    """
    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(os.path.dirname(VECTOR_STORE_DIR), exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_DIR)
    print(f"Vector store saved to {VECTOR_STORE_DIR}")
    return vectorstore

def load_vector_store():
    """
    Loads the FAISS vector store from disk.
    """
    if not os.path.exists(VECTOR_STORE_DIR):
        print("Vector store not found. Need to build it first.")
        return None
        
    embeddings = get_embeddings_model()
    # allow_dangerous_deserialization is required for FAISS local loading in newer langchain versions
    vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore
