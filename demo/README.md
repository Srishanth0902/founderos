# Demo recording

`founderos_demo.mp4` is a 1:30 narrated walkthrough of FounderOS, and [SCRIPT.md](SCRIPT.md) has the 1:00 presenter script, the scene list and the sample cases. This folder can re-record it.

| File | Purpose |
|---|---|
| `record_founderos.py` | Starts the app, drives it in a real browser, records the narrated video |
| `recorder.py` | Reusable parts: cursor, highlights, cards, TTS narration, screen capture, sync, encoding |
| `app_demo.py` | Streamlit entry point that runs `founderos/app.py` unchanged and logs every tool call |
| `../founderos/offline_model.py` | Offline stand-in for the Gemini chat model and embeddings, used only when no API key is set |

## Running the app without an API key

```bash
streamlit run demo/app_demo.py
```

When no `GOOGLE_API_KEY`, `GEMINI_API_KEY` or `OPENAI_API_KEY` is set (in the environment or `founderos/.env`),
`app_demo.py` swaps in `founderos/offline_model.py` and builds a FAISS index with local embeddings in
`founderos/data/vector_store_offline/`. The stand-in is not a language model. It chooses tools with keyword rules,
LangGraph executes them, and it builds answers only from the tool results. The sidebar says so. With a key set, the
real model and the normal index (`python init_rag.py`) are used.

## Re-recording

Requirements: Linux, `Xvfb`, `ffmpeg` built with libass, Chromium, FounderOS's own requirements, and `requirements.txt` here.

```bash
pip install -r founderos/requirements.txt -r demo/requirements.txt
mkdir -p ~/.tts && cd ~/.tts
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd -

python demo/record_founderos.py      # add GOOGLE_API_KEY to founderos/.env to record with Gemini
```

Settings (environment variables): `FOUNDEROS_PORT` (8599), `STREAMLIT_BIN` (`streamlit`), `DEMO_CHROME`
(`/opt/pw-browsers/chromium`), `DEMO_DISPLAY` (`:99`), `KOKORO_DIR` (`~/.tts`), `DEMO_VOICE` / `DEMO_SPEED`
(`af_heart` / `1.0`), `DEMO_SECONDS` (90), `DEMO_WORK_DIR` (`demo/.work`).
