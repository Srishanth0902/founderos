"""
FounderOS web app for serverless hosting such as Vercel, where Streamlit cannot run.

It serves the same LangGraph agent as app.py (same prompts, tools and FAISS knowledge base)
behind a small FastAPI API and a single-page chat UI (static/index.html).

    cd founderos && uvicorn web:app --reload

Without GOOGLE_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY it runs the offline stand-in
model in offline_model.py, and the page says so.
"""
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

OFFLINE = not any(os.getenv(k) for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"))
ON_VERCEL = bool(os.getenv("VERCEL") or os.getenv("__VC_HANDLER_ENTRYPOINT"))

app = FastAPI(title="FounderOS")
_agent = None
_agent_lock = threading.Lock()


def get_agent():
    """Build the LangGraph agent on first use (this also builds the FAISS index if it is missing)."""
    global _agent
    with _agent_lock:
        if _agent is None:
            if OFFLINE:
                import offline_model

                offline_model.install()
                os.environ["GOOGLE_API_KEY"] = "offline-stand-in"
                # Keep the local-embedding index apart from a Gemini-embedding one.
                base = tempfile.gettempdir() if ON_VERCEL else str(BASE_DIR / "data")
                os.environ.setdefault("VECTOR_STORE_DIR", os.path.join(base, "vector_store_offline"))
            from agent.graph import create_agent_graph

            _agent = create_agent_graph()
        return _agent


def text_of(content: Any) -> str:
    """Flatten a chat model's content (a string, or a list of text parts) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(part for part in (text_of(item) for item in content) if part)
    if isinstance(content, dict):
        return text_of(content.get("text", content.get("content", "")))
    return str(getattr(content, "text", content))


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    mode: Literal["normal", "recruiter", "daily_assistant"] = "normal"
    history: List[Turn] = Field(default_factory=list, max_length=40)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/info")
def info() -> dict:
    if OFFLINE:
        model = "offline stand-in (no API key)"
    elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        model = os.getenv("CHAT_MODEL", "gemini-flash-latest")
    else:
        model = os.getenv("CHAT_MODEL", "gpt-4o")
    return {"offline": OFFLINE, "model": model}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    try:
        graph = get_agent()
    except Exception as err:  # e.g. an invalid API key while building the knowledge index
        raise HTTPException(status_code=503, detail=f"The agent could not start: {err}")

    # Each request carries its own history, so any serverless instance can answer it.
    messages = [HumanMessage(t.content) if t.role == "user" else AIMessage(t.content) for t in req.history[-20:]]
    messages.append(HumanMessage(req.message))
    try:
        result = graph.invoke({"messages": messages, "mode": req.mode},
                              config={"configurable": {"thread_id": uuid.uuid4().hex}})
    except Exception as err:
        raise HTTPException(status_code=502, detail=f"The model call failed: {err}")

    new = result["messages"][len(messages):]
    return {
        "answer": text_of(getattr(result["messages"][-1], "content", "")),
        "tools": [m.name for m in new if isinstance(m, ToolMessage)],
    }
