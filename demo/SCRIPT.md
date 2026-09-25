# FounderOS demo: scripts and walkthrough

Files in this folder:

- `founderos_demo.mp4`: 1:30 narrated screen recording of the app (1920×1080, burned-in captions)
- `founderos_demo.srt`: the same captions as a subtitle file
- `record_founderos.py`, `recorder.py`, `app_demo.py`, `offline_model.py`: how it was produced (see [README.md](README.md))

> **About the model in this recording.** No working Gemini or OpenAI key was available in the recording environment, so the
> video runs FounderOS with an offline stand-in for the chat model (`offline_model.py`). Everything else is the real app:
> `app.py`, the LangGraph graph, the tools and a FAISS index over `founderos/knowledge/`. The stand-in picks tools with
> keyword rules. LangGraph executes those tool calls, and the stand-in builds each answer only from what the tools
> returned. The sidebar and the closing card say this on screen. With a key in `founderos/.env`, the same recorder uses
> Gemini and drops the note.

## 1. The 1:00 script

Written for Srishanth to present in the first person. About 135 words, 55 to 60 seconds.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:08 | App home | "Recruiters skim resumes, clients ask the same questions, and founders juggle everything. So I built FounderOS, my AI digital twin." |
| 0:08–0:20 | Sidebar modes and suggested actions | "It's a LangGraph agent over a FAISS knowledge base of my projects, skills, beliefs and routines, with tools for calculations, web search, planning and business advice." |
| 0:20–0:33 | Normal Mode: **Learn about me**, then **Calculate ROI** | "In Normal Mode it answers as me. Ask about my background and it retrieves my real profile. Ask for an ROI and the calculator does the maths: two hundred percent." |
| 0:33–0:50 | Recruiter Mode question, then Daily Assistant Mode plan | "Recruiter Mode pitches my projects and stack to hiring managers. Daily Assistant Mode plans my day around my actual routine, and the project planner turns an idea into an architecture, a timeline and risks." |
| 0:50–1:00 | Business-advice answer | "It never invents achievements. It only answers from what I've actually done. FounderOS: an AI co-founder that thinks like me." |

## 2. The 1:30 video, scene by scene

Timestamps are from the final recording.

| Time | Scene | Narration |
|---|---|---|
| 0:00 | Title card | Meet FounderOS, Srishanth's AI digital twin. It knows his experience, remembers his projects, and reasons like him. |
| 0:08 | Home screen, "Inside the agent" panel, mode selector | Under the hood it's a LangGraph agent with a FAISS knowledge base of his projects, skills, beliefs and routines, plus tools for maths, web search, planning and business advice. The sidebar switches between three modes. |
| 0:22 | **Learn about me** with its agent-trace panel | Ask it to introduce itself, and the agent calls the knowledge retriever, then answers in first person from his real profile. |
| 0:30 | **Calculate ROI** | Business maths goes to the calculator tool: five hundred dollars that returns fifteen hundred is a two hundred percent ROI. |
| 0:39 | Recruiter Mode question | In Recruiter Mode, the same agent pitches Srishanth to a hiring manager: his projects, his tech stack, and his philosophy, in his own words. |
| 0:48 | Daily Assistant Mode: "Plan my day…" | Daily Assistant Mode plans his day around his real routine: learning in the morning, building in the afternoon, and proposals in the evening. |
| 0:58 | **Plan a project** | And for a new idea, the project planner drafts the architecture, timeline, risks and deployment, using his preferred stack. |
| 1:06 | Normal Mode: client-project advice question | Back in Normal Mode, ask whether to take a well-paid client project in an unfamiliar stack, and it reasons through his own decision framework, not generic tips. |
| 1:17 | Closing card | FounderOS: an AI co-founder that thinks like its founder. This recording uses an offline stand-in model; add a Gemini key for live answers. |

The agent-trace panel at the bottom left of each answer comes from a log of the tool calls that actually ran. It is not scripted.

## 3. Sample cases shown

| Mode | Question | Tools the agent called | Answer (from the knowledge base or tool output) |
|---|---|---|---|
| Normal | "Tell me about yourself." (Learn about me) | `founder_knowledge_retriever` | First-person intro from `interview_answers.md`, current focus (LangChain, AI Agents, RAG, automation…), what he builds, mission, 5-step working style |
| Normal | "Calculate the ROI of a $500 investment that yields $1500." (Calculate ROI) | `calculator("(1500 - 500) / 500 * 100")` returned `200.0` | ROI = 200%: invested $500, returned $1,500, net gain $1,000, a 3x return |
| Recruiter | "What has Srishanth built, and what tech stack does he use?" | `founder_knowledge_retriever` × 2 | SkillVerse, AI Automation Workflows, Founder Digital Twin, each with its tech; quick-facts stack (Python, React, n8n, LangChain, Vercel); his quote on projects over certificates |
| Daily Assistant | "Plan my day: finish the FounderOS demo video, write a client proposal, and learn LangGraph." | `daily_planner`, `founder_knowledge_retriever` | Morning: learn LangGraph. Afternoon: finish the demo video. Evening: write the proposal. Each slot is merged with his routine from `daily_routine.md`, followed by his priority order |
| Daily Assistant | "Help me plan a new AI SaaS project." (Plan a project) | `project_planner`, `founder_knowledge_retriever` × 2 | Architecture from his preferred stack (React/Next.js + Tailwind, FastAPI/Flask, LangChain/LangGraph + FAISS, n8n), his build loop as the timeline, risks, deployment on Vercel/Render |
| Normal | "Should I take a client project that pays well but uses a stack I don't know?" | `founder_advisor`, `founder_knowledge_retriever` | His decision checklist from `decision_framework.md`, how he selects projects and technology, and a verdict rule |

## 4. Features covered

| Feature | Where in the video |
|---|---|
| LangGraph agent with a tool loop and memory checkpointer | 0:08 (panel), every answer (trace) |
| RAG over the founder's knowledge base (FAISS, top-5 chunks) | 0:22, 0:39, 0:48, 0:58, 1:06 |
| Calculator tool | 0:30 |
| Three modes (Normal, Recruiter, Daily Assistant) with their own system prompts | 0:08, 0:39, 0:48, 1:06 |
| Suggested actions in the sidebar | 0:22, 0:30, 0:58 |
| Planners and advisor tools | 0:48, 0:58, 1:06 |

The web search tool (`web_search`, DuckDuckGo) is part of the agent but is not shown, because the recording environment could not reach DuckDuckGo.
