"""
Streamlit entry point for recording the FounderOS demo.

    streamlit run demo/app_demo.py

It runs founderos/app.py unchanged. Two things are added around it:

* No GOOGLE_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY (or FOUNDEROS_OFFLINE=1):
  the Gemini classes are replaced by the offline stand-in in founderos/offline_model.py and
  a FAISS index is built with local embeddings in data/vector_store_offline.
  With a key, the real model and the normal index (python init_rag.py) are used.
* Every tool call (name, input, output) is appended to the JSONL file named by
  FOUNDEROS_TOOL_LOG, so the recorder can show which tools actually ran.
"""
import json
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(os.path.dirname(HERE), "founderos")
sys.path[:0] = [APP_DIR, HERE]
os.chdir(APP_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(APP_DIR, ".env"))
HAS_KEY = any(os.getenv(k) for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"))
OFFLINE = (os.getenv("FOUNDEROS_OFFLINE") == "1" or not HAS_KEY
           or os.getenv("GOOGLE_API_KEY") == "offline-stand-in")  # set below; persists across reruns

if OFFLINE and "config" not in sys.modules:
    import offline_model

    offline_model.install()
    os.environ["GOOGLE_API_KEY"] = "offline-stand-in"
    import config

    config.VECTOR_STORE_DIR = "data/vector_store_offline"
    if not os.path.exists(config.VECTOR_STORE_DIR):
        from rag.loader import load_documents
        from rag.splitter import split_documents
        from rag.vectorstore import build_vector_store

        build_vector_store(split_documents(load_documents(config.KNOWLEDGE_DIR)))

import agent.nodes as nodes  # noqa: E402

TOOL_LOG = os.getenv("FOUNDEROS_TOOL_LOG")
if TOOL_LOG and not getattr(nodes, "_demo_logging", False):
    def _logged(func, name):
        def wrapper(*args, **kwargs):
            started = time.time()
            result = func(*args, **kwargs)
            with open(TOOL_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": started, "tool": name, "input": kwargs or list(args),
                                     "output": str(result)[:4000], "ms": round((time.time() - started) * 1000, 1)}) + "\n")
            return result
        return wrapper

    for tool in nodes.tools:
        if getattr(tool, "func", None):
            tool.func = _logged(tool.func, tool.name)
    nodes._demo_logging = True

runpy.run_path(os.path.join(APP_DIR, "app.py"), run_name="__main__")

if OFFLINE:
    import streamlit as st

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Demo model:** offline stand-in (no API key configured). The LangGraph agent, "
        "tools and FAISS retrieval run for real. Answers are assembled only from tool "
        "results. Add GOOGLE_API_KEY to use Gemini."
    )
