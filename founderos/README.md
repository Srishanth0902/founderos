# FounderOS – My Digital Twin

An AI-powered digital twin that understands my experience, remembers my projects, reasons like me, and helps recruiters, clients, and myself solve real-world problems.

## Setup Instructions

1. **Install Python Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Open the `.env` file and add your `GOOGLE_API_KEY` or `OPENAI_API_KEY`.

3. **Initialize Knowledge Base (optional):**
   The app builds the FAISS vector database from the knowledge files on first start. To build it ahead of time:
   ```bash
   python init_rag.py
   ```

4. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## Deployment

### Vercel (web version)

Vercel cannot run Streamlit, so `web.py` serves the same agent (prompts, tools, FAISS knowledge base) through FastAPI
with a single-page chat UI in `static/index.html`. `pyproject.toml` and `vercel.json` in this folder configure it.

1. In Vercel, import the repository and set **Root Directory** to `founderos`.
2. Add `GOOGLE_API_KEY` under Settings, then Environment Variables. Without a key the site runs in a labelled offline
   demo mode that answers only from the knowledge base and tool results.
3. Deploy. The first chat on a new instance builds the FAISS index in `/tmp` (a few seconds). Vercel also installs part
   of the dependencies when an instance starts, because together they exceed its bundle size limit.

Run the web version locally with `uvicorn web:app --reload` from this folder.

### Streamlit

**Streamlit Community Cloud (free):**
1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and choose **Create app**, then the option to deploy from a GitHub repo.
2. Repository `Srishanth0902/capabl_hack`, branch `main`, main file path `founderos/app.py`.
3. **Advanced settings**: Python 3.12, and under **Secrets** add:
   ```toml
   GOOGLE_API_KEY = "your-gemini-api-key"
   ```
4. Deploy. The first start builds the FAISS index from `founderos/knowledge/` (one embeddings request), then the app is
   live at `https://<your-app-name>.streamlit.app`.

Any host that runs a persistent process (Render, Railway, Hugging Face Spaces) also works:
`streamlit run founderos/app.py --server.port $PORT --server.address 0.0.0.0`

Optional settings: `CHAT_MODEL` (default `gemini-flash-latest`) and `EMBEDDING_MODEL` (default `models/gemini-embedding-001`).

## Architecture
- **UI:** Streamlit Chat UI
- **Agent:** LangGraph with memory and mode switching
- **RAG:** FAISS + LangChain Embeddings
- **Tools:** Calculator, DuckDuckGo Search, Custom Prompts for Founder Advisor, Resume Coach, and Planners.
