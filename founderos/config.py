import os
from dotenv import load_dotenv

load_dotenv()

# We'll default to Gemini but fallback to OpenAI if needed
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_PROVIDER = "gemini" if GEMINI_API_KEY else "openai" if OPENAI_API_KEY else None

if not LLM_PROVIDER:
    raise ValueError("Please set GOOGLE_API_KEY or OPENAI_API_KEY in the .env file.")

# RAG Configurations
KNOWLEDGE_DIR = "knowledge"
VECTOR_STORE_DIR = "data/vector_store"

# Embedding Model
if LLM_PROVIDER == "gemini":
    EMBEDDING_MODEL = "models/embedding-001"
    CHAT_MODEL = "gemini-2.5-flash"
else:
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o"
