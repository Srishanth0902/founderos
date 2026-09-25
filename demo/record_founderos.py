"""
Record the narrated FounderOS demo (about 1:30).

    python demo/record_founderos.py

Starts FounderOS through demo/app_demo.py on a free port, drives it in a real
browser, and writes demo/founderos_demo.mp4 and demo/founderos_demo.srt.
Without GOOGLE_API_KEY / OPENAI_API_KEY the offline stand-in model is used and
the video says so. See demo/README.md.
"""
import asyncio
import html
import json
import os
import subprocess
import sys
import time
import urllib.request

from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recorder import Ctx, Narrator, Recorder, Segment, card_html  # noqa: E402

PORT = int(os.getenv("FOUNDEROS_PORT", "8599"))
URL = f"http://localhost:{PORT}"
WORK = os.getenv("DEMO_WORK_DIR", os.path.join(HERE, ".work"))
TOOL_LOG = os.path.join(WORK, "founderos_tools.jsonl")
ACCENT = "#c4b5fd"
BG = "radial-gradient(1200px 600px at 20% 10%, #4c1d95 0%, #0b1220 58%)"
MSG = '[data-testid="stChatMessage"]'
SCROLLER = '[data-testid="stAppScrollToBottomContainer"]'
TRACE_POS = {"left": "12px", "bottom": "12px", "width": "276px", "maxWidth": "276px"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tool_calls(since: int) -> list:
    if not os.path.exists(TOOL_LOG):
        return []
    with open(TOOL_LOG, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh.readlines()[since:]]


def log_size() -> int:
    if not os.path.exists(TOOL_LOG):
        return 0
    with open(TOOL_LOG, encoding="utf-8") as fh:
        return len(fh.readlines())


def trace_html(calls: list) -> str:
    rows = []
    for c in calls:
        arg = next(iter(c["input"].values()), "") if isinstance(c["input"], dict) else ""
        arg = arg if len(arg) <= 46 else arg[:44] + "…"
        if c["tool"] == "founder_knowledge_retriever":
            result = "top-5 FAISS chunks"
        elif c["tool"] == "calculator":
            result = f"<code>{html.escape(c['output'])}</code>"
        else:
            result = "planning brief"
        rows.append(f'<div class="row"><b>{c["tool"]}</b><br><code>{html.escape(arg)}</code> → {result}</div>')
    return ('<div style="font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8">'
            f'Agent trace · {len(calls)} tool call{"s" if len(calls) != 1 else ""}</div>' + "".join(rows))


async def ask(ctx: Ctx, prompt: str = "", button: str = "") -> list:
    """Send a prompt (typed, or via a suggested-action button) and wait for the answer."""
    before_msgs = await ctx.page.locator(MSG).count()
    before_log = log_size()
    if button:
        await ctx.click(f'[data-testid="stSidebar"] button:has-text("{button}")')
    else:
        await ctx.type('[data-testid="stChatInputTextArea"]', prompt, delay=0.022)
        await asyncio.sleep(0.2)
        await ctx.page.keyboard.press("Enter")
    await ctx.page.wait_for_function(
        f"document.querySelectorAll('{MSG}').length >= {before_msgs + 2} && !document.body.innerText.includes('FounderOS is thinking')",
        timeout=120000)
    await asyncio.sleep(0.6)
    return tool_calls(before_log)


async def show_answer(ctx: Ctx) -> None:
    """Scroll so the latest question sits at the top of the view."""
    await ctx.page.evaluate(
        f"""() => {{ const box = document.querySelector('{SCROLLER}'); const msgs = document.querySelectorAll('{MSG}');
        const q = msgs[msgs.length - 2]; if (!box || !q) return;
        const top = q.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 24;
        box.scrollTo({{top, behavior: 'smooth'}}); }}""")
    await asyncio.sleep(0.8)


async def scroll_by(ctx: Ctx, px: int) -> None:
    await ctx.page.evaluate(f"() => document.querySelector('{SCROLLER}').scrollBy({{top: {px}, behavior: 'smooth'}})")
    await asyncio.sleep(0.8)


async def set_mode(ctx: Ctx, mode: str) -> None:
    await ctx.click('[data-testid="stSelectbox"]')
    await asyncio.sleep(0.35)
    await ctx.click(f'[role="option"]:has-text("{mode}")', duration=0.4)
    await ctx.page.wait_for_function(
        """(mode) => { const el = document.querySelector('[data-testid="stSelectbox"]');
        return el.textContent.includes(mode) || [...el.querySelectorAll('input')].some((i) => (i.value || '').includes(mode)); }""",
        arg=mode, timeout=10000)
    await asyncio.sleep(0.8)


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------
async def title(ctx: Ctx) -> None:
    await ctx.frac(1.0)
    await ctx.page.evaluate("window.__demo.hideCards()")


async def overview(ctx: Ctx) -> None:
    await ctx.at(0.4)
    await ctx.callout(
        '<div style="font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8">Inside the agent</div>'
        '<div class="row"><b>LangGraph</b> agent ⇄ ToolNode loop with conversation memory</div>'
        '<div class="row"><b>RAG</b>: FAISS index over the founder\'s knowledge files</div>'
        '<div class="row"><b>Tools</b>: <code>founder_knowledge_retriever</code>, <code>calculator</code>, '
        '<code>web_search</code>, <code>project_planner</code>, <code>daily_planner</code>, '
        '<code>resume_coach</code>, <code>founder_advisor</code></div>',
        {"left": "620px", "top": "300px", "maxWidth": "520px"})
    await ctx.cue(1)
    await ctx.highlight('[data-testid="stSelectbox"]', "Three modes", below=True)
    await ctx.frac(0.9)
    await ctx.clear()


async def about(ctx: Ctx) -> None:
    await ctx.highlight('[data-testid="stSidebar"] button:has-text("Learn about me")', "Suggested action")
    await asyncio.sleep(0.6)
    await ctx.clear()
    calls = await ask(ctx, button="Learn about me")
    await show_answer(ctx)
    await ctx.callout(trace_html(calls), TRACE_POS)
    await ctx.frac(0.75)
    await scroll_by(ctx, 220)


async def roi(ctx: Ctx) -> None:
    await ctx.clear()
    calls = await ask(ctx, button="Calculate ROI")
    await show_answer(ctx)
    await ctx.callout(trace_html(calls), TRACE_POS)


async def recruiter(ctx: Ctx) -> None:
    await ctx.clear()
    await set_mode(ctx, "Recruiter Mode")
    calls = await ask(ctx, "What has Srishanth built, and what tech stack does he use?")
    await show_answer(ctx)
    await ctx.callout(trace_html(calls), TRACE_POS)
    await ctx.frac(0.8)
    await scroll_by(ctx, 260)


async def daily(ctx: Ctx) -> None:
    await ctx.clear()
    await set_mode(ctx, "Daily Assistant Mode")
    calls = await ask(ctx, "Plan my day: finish the FounderOS demo video, write a client proposal, and learn LangGraph.")
    await show_answer(ctx)
    await ctx.callout(trace_html(calls), TRACE_POS)
    await ctx.frac(0.82)
    await scroll_by(ctx, 240)


async def project(ctx: Ctx) -> None:
    await ctx.clear()
    calls = await ask(ctx, button="Plan a project")
    await show_answer(ctx)
    await ctx.callout(trace_html(calls), TRACE_POS)
    await ctx.frac(0.55)
    await scroll_by(ctx, 300)


async def advice(ctx: Ctx) -> None:
    await ctx.clear()
    await set_mode(ctx, "Normal Mode")
    calls = await ask(ctx, "Should I take a client project that pays well but uses a stack I don't know?")
    await show_answer(ctx)
    await ctx.callout(trace_html(calls), TRACE_POS)


async def outro(ctx: Ctx) -> None:
    await ctx.clear()
    await ctx.page.evaluate("([h, b]) => window.__demo.card(h, b)", [OUTRO, BG])


def is_offline() -> bool:
    """Same rule as app_demo.py: the stand-in is used unless an API key is configured."""
    from dotenv import dotenv_values
    keys = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")
    dotenv = dotenv_values(os.path.join(os.path.dirname(HERE), "founderos", ".env"))
    return os.getenv("FOUNDEROS_OFFLINE") == "1" or not any(os.getenv(k) or dotenv.get(k) for k in keys)


OFFLINE = is_offline()

TITLE = card_html(
    "Digital twin agent", "FounderOS",
    "An AI digital twin that understands Srishanth's experience, remembers his projects, and reasons like him, "
    "for recruiters, clients, and his own day.",
    ["LangGraph agent", "FAISS RAG", "7 tools", "3 modes", "Streamlit"], ACCENT)
OUTRO_NOTE = ("Recorded with an offline stand-in model (no API key in this environment). "
              "The LangGraph agent, tool calls and FAISS retrieval are real, and the answers are built only from their results.")
OUTRO = card_html(
    "FounderOS", "An AI co-founder<br>that thinks like its founder.",
    "Answers from what he has actually done. It never invents achievements.",
    ["streamlit run app.py", "Gemini or OpenAI"], ACCENT, OUTRO_NOTE if OFFLINE else "")

SEGMENTS = [
    Segment("title", "Meet FounderOS, Srishanth's AI digital twin. It knows his experience, remembers his projects, "
                     "and reasons like him.", title, hold=0.3),
    Segment("overview", "Under the hood it's a LangGraph agent with a FAISS knowledge base of his projects, skills, beliefs "
                        "and routines, plus tools for maths, web search, planning and business advice. "
                        "The sidebar switches between three modes.", overview),
    Segment("about", "Ask it to introduce itself, and the agent calls the knowledge retriever, then answers in first person "
                     "from his real profile.", about),
    Segment("roi", "Business maths goes to the calculator tool: five hundred dollars that returns fifteen hundred "
                   "is a two hundred percent ROI.", roi),
    Segment("recruiter", "In Recruiter Mode, the same agent pitches Srishanth to a hiring manager: his projects, his tech stack, "
                         "and his philosophy, in his own words.", recruiter),
    Segment("daily", "Daily Assistant Mode plans his day around his real routine: learning in the morning, building in the "
                     "afternoon, and proposals in the evening.", daily),
    Segment("project", "And for a new idea, the project planner drafts the architecture, timeline, risks and deployment, "
                       "using his preferred stack.", project),
    Segment("advice", "Back in Normal Mode, ask whether to take a well-paid client project in an unfamiliar stack, "
                      "and it reasons through his own decision framework, not generic tips.", advice),
    Segment("outro", "FounderOS: an AI co-founder that thinks like its founder."
                     + (" This recording uses an offline stand-in model; add a Gemini key for live answers." if OFFLINE else ""),
            outro, hold=1.0),
]


def start_app() -> subprocess.Popen:
    os.makedirs(WORK, exist_ok=True)
    if os.path.exists(TOOL_LOG):
        os.remove(TOOL_LOG)
    env = dict(os.environ, FOUNDEROS_TOOL_LOG=TOOL_LOG)
    streamlit = os.getenv("STREAMLIT_BIN", "streamlit")
    proc = subprocess.Popen(
        [streamlit, "run", os.path.join(HERE, "app_demo.py"), "--server.headless", "true", "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false", "--theme.base", "light", "--client.toolbarMode", "minimal"],
        env=env, stdout=open(os.path.join(WORK, "streamlit.log"), "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen(URL + "/_stcore/health", timeout=2)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("FounderOS did not start; see .work/streamlit.log")


async def main() -> None:
    narrator = Narrator(os.path.join(WORK, "tts"), voice=os.getenv("DEMO_VOICE", "af_heart"), speed=float(os.getenv("DEMO_SPEED", "1.0")))
    rec = Recorder("founderos_demo", HERE, WORK, narrator, target_seconds=float(os.getenv("DEMO_SECONDS", "90")))
    rec.prepare(SEGMENTS)
    app = start_app()
    try:
        rec.start_display()
        async with async_playwright() as p:
            browser, context = await rec.launch(p)
            page = await context.new_page()
            await rec.fullscreen(page)
            await page.goto(URL)
            await page.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=120000)
            await page.wait_for_timeout(2500)
            await rec.perform(page, SEGMENTS, TITLE, BG)
            await browser.close()
        rec.close()
    finally:
        app.terminate()
    rec.finish(SEGMENTS)


if __name__ == "__main__":
    asyncio.run(main())
