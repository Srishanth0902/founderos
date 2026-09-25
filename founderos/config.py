import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# Paths are relative to this file, so the app works whatever directory it is started from
# (hosting platforms such as Streamlit Community Cloud run it from the repository root).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# We'll default to Gemini but fallback to OpenAI if needed
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if GEMINI_API_KEY and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY  # langchain-google-genai reads GOOGLE_API_KEY

LLM_PROVIDER = "gemini" if GEMINI_API_KEY else "openai" if OPENAI_API_KEY else None

if not LLM_PROVIDER:
    raise ValueError("Please set GOOGLE_API_KEY or OPENAI_API_KEY in the .env file.")

# RAG Configurations
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
# Serverless hosts (Vercel) only allow writing to the temp directory.
_ON_VERCEL = bool(os.getenv("VERCEL") or os.getenv("__VC_HANDLER_ENTRYPOINT"))
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR") or (
    os.path.join(tempfile.gettempdir(), "founderos_vector_store") if _ON_VERCEL
    else os.path.join(BASE_DIR, "data", "vector_store"))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

# Models (override with CHAT_MODEL / EMBEDDING_MODEL). The Gemini defaults are current names:
# "gemini-flash-latest" always points at the newest Flash model, and embedding-001 has been retired.
if LLM_PROVIDER == "gemini":
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-flash-latest")
else:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
