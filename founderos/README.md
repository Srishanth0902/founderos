# FounderOS – My Digital Twin

An AI-powered digital twin that understands my experience, remembers my projects, reasons like me, and helps recruiters, clients, and myself solve real-world problems.

## Setup Instructions

1. **Install Python Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Open the `.env` file and add your `GOOGLE_API_KEY` or `OPENAI_API_KEY`.

3. **Initialize Knowledge Base:**
   Before running the app, you need to build the FAISS vector database from the knowledge files.
   ```bash
   python init_rag.py
   ```

4. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## Architecture
- **UI:** Streamlit Chat UI
- **Agent:** LangGraph with memory and mode switching
- **RAG:** FAISS + LangChain Embeddings
- **Tools:** Calculator, DuckDuckGo Search, Custom Prompts for Founder Advisor, Resume Coach, and Planners.
