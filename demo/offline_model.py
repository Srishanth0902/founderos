"""
Offline stand-in for the Gemini chat model and embeddings, for recording demos
without an API key.

FounderOS's own code runs unchanged: config.py, the LangGraph graph, the tools
and the FAISS retriever. install() only replaces the `langchain_google_genai`
module, so `ChatGoogleGenerativeAI` becomes OfflineFounderModel and
`GoogleGenerativeAIEmbeddings` becomes LocalHashEmbeddings.

OfflineFounderModel is not a language model. It picks tools with keyword rules,
emits real tool calls that LangGraph executes, and then assembles its answer
from what the tools returned: retrieved knowledge-base chunks, calculator
output, and so on. It never adds facts that the tools did not return.
"""
import hashlib
import json
import re
import sys
import types
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

MODEL_LABEL = "offline stand-in (no API key)"

# ---------------------------------------------------------------------------
# Embeddings: hashed unigrams + bigrams, no download, deterministic
# ---------------------------------------------------------------------------
_STOP = set(
    "a an the and or of to in on for with is are was were be been being i me my mine you your yours he his "
    "she her it its this that these those what which who whom how do does did can could would should will "
    "about tell from at by as into than then so if not no yes we our they them their there here have has had "
    "just also more most very really any some all each other such only own same too s t don now please help "
    "srishanth boddula founderos".split()
)
_DIM = 2048


def _stem(word: str) -> str:
    if len(word) > 4:
        return re.sub(r"(ies|es|s|ing|ed)$", "", word)
    return word


def _terms(text: str) -> List[str]:
    words = [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def _embed(text: str) -> List[float]:
    vec = np.zeros(_DIM, dtype=np.float32)
    for term in _terms(text):
        vec[int(hashlib.md5(term.encode()).hexdigest(), 16) % _DIM] += 1.0
    vec = np.log1p(vec)
    norm = float(np.linalg.norm(vec))
    return (vec / norm if norm else vec).tolist()


class LocalHashEmbeddings(Embeddings):
    def __init__(self, model: Optional[str] = None, **_: Any) -> None:
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return _embed(text)


# ---------------------------------------------------------------------------
# Knowledge-base text parsing (works on whatever chunks the retriever returned)
# ---------------------------------------------------------------------------
_FRONT_MATTER = re.compile(r"^(document_type|owner|topics|version):")
_BULLET = re.compile(r"^(?:[-*•]|\d+[.)])\s+")


def parse_blocks(text: str) -> List[Dict[str, Any]]:
    """Split markdown chunks into blocks: {top, heading, label, items, paras}."""
    lines = [l.strip() for l in text.replace("\r", "").split("\n")]
    blocks: List[Dict[str, Any]] = []
    top = ""

    def new(heading: str, label: Optional[str] = None) -> Dict[str, Any]:
        return {"top": top, "heading": heading, "label": label, "items": [], "paras": []}

    cur = new("")
    in_front = False
    for i, line in enumerate(lines):
        if line == "---":
            nxt = next((l for l in lines[i + 1:] if l), "")
            in_front = bool(_FRONT_MATTER.match(nxt))
            continue
        if not line or line in {"↓"}:
            continue
        if in_front or _FRONT_MATTER.match(line):
            continue
        if line.startswith("#"):
            blocks.append(cur)
            level = len(line) - len(line.lstrip("#"))
            heading = line.lstrip("#").strip()
            if level == 1:
                top = heading
            cur = new(heading)
        elif _BULLET.match(line):
            cur["items"].append(_BULLET.sub("", line))
        else:
            nxt = next((l for l in lines[i + 1:] if l), "")
            if _BULLET.match(nxt) and len(line) < 40 and not line.endswith((".", ":")):
                blocks.append(cur)
                cur = new(cur["heading"], label=line)
            else:
                cur["paras"].append(line)
    blocks.append(cur)
    kept = [b for b in blocks if b["heading"] and (b["items"] or b["paras"])]
    if kept:
        kept[0]["raw"] = text
    return kept


def kb_raw(blocks: List[Dict[str, Any]]) -> str:
    return blocks[0].get("raw", "") if blocks else ""


def find(blocks: List[Dict[str, Any]], *names: str, label: Optional[str] = None) -> Optional[Dict[str, Any]]:
    wanted = [n.lower() for n in names]
    for b in blocks:
        if b["heading"].lower() in wanted and (label is None or (b["label"] or "").lower() == label.lower()):
            return b
    return None


def first_sentences(paras: List[str], n: int = 2) -> str:
    text = " ".join(paras)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n]).strip()


def join_items(items: List[str], limit: int = 6) -> str:
    return ", ".join(items[:limit])


# ---------------------------------------------------------------------------
# Intent routing
# ---------------------------------------------------------------------------
def detect_mode(messages: List[Any]) -> str:
    for m in messages:
        if isinstance(m, SystemMessage):
            text = str(m.content)
            if "Recruiter Mode" in text:
                return "recruiter"
            if "Daily Assistant Mode" in text:
                return "daily"
    return "normal"


def detect_intent(text: str) -> str:
    t = text.lower()
    if re.search(r"\broi\b|calculate|\bprofit\b|\bmargin\b|percent|%|\d\s*[-+*/x]\s*\d", t):
        return "calc"
    if re.search(r"plan (my|the) day|my day|schedule|today", t):
        return "daily_plan"
    if re.search(r"\bplan\b", t) and re.search(r"project|saas|app|product|build|startup", t):
        return "project_plan"
    if re.search(r"resume|strength|weakness|interview|star stor", t):
        return "resume"
    if re.search(r"should i|pricing|price|charge|client|market|proposal", t):
        return "advisor"
    if re.search(r"latest|news|this week|trending", t):
        return "web"
    if re.search(r"\b(hi|hello|hey)\b", t) and len(t.split()) <= 3:
        return "greeting"
    if re.search(r"project|built|build|portfolio", t) or re.search(r"stack|technolog|tools? (do|does)|framework", t):
        return "profile_work"
    return "profile"


def extract_expression(text: str) -> Optional[str]:
    t = text.lower().replace(",", "")
    nums = [float(n) for n in re.findall(r"\$?\s*(\d+(?:\.\d+)?)", t)]
    if "roi" in t and len(nums) >= 2:
        cost, value = nums[0], nums[1]
        return f"({value:g} - {cost:g}) / {cost:g} * 100"
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", t)
    if m:
        return f"{m.group(1)} / 100 * {m.group(2)}"
    m = re.search(r"[\d.\s()+\-*/]{3,}", t)
    if m and re.search(r"\d\s*[-+*/]\s*\d", m.group(0)):
        return m.group(0).strip()
    return None


def extract_tasks(text: str) -> List[str]:
    body = re.split(r"plan (?:my|the) day[:,.]?|today[:,.]?", text, maxsplit=1, flags=re.I)[-1]
    body = re.sub(r"^\s*(i need to|i have to|i want to|help me)\s*", "", body, flags=re.I)
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+(?=[a-z]+\s)", body.strip().rstrip("."))
    return [p.strip().rstrip(".") for p in parts if len(p.strip()) > 3][:5]


def tool_call(name: str, **args: Any) -> Dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}


# ---------------------------------------------------------------------------
# The stand-in chat model
# ---------------------------------------------------------------------------
class OfflineFounderModel(BaseChatModel):
    model: str = "offline-founder"
    temperature: float = 0.0
    bound_tools: List[str] = []

    @property
    def _llm_type(self) -> str:
        return "offline-founder-standin"

    def bind_tools(self, tools: Any, **_: Any) -> "OfflineFounderModel":
        names = [getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools]
        return self.model_copy(update={"bound_tools": names})

    # --- plumbing -----------------------------------------------------------
    def _generate(self, messages: List[Any], stop: Optional[List[str]] = None, run_manager: Any = None, **_: Any) -> ChatResult:
        last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
        question = str(messages[last_human].content) if last_human >= 0 else ""
        results = [m for m in messages[last_human + 1:] if isinstance(m, ToolMessage)]
        mode = detect_mode(messages)
        if results:
            message = AIMessage(content=self._answer(question, mode, results))
        else:
            calls = self._plan(question, mode)
            message = AIMessage(content="", tool_calls=calls) if calls else AIMessage(content=self._answer(question, mode, []))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _has(self, name: str) -> bool:
        return not self.bound_tools or name in self.bound_tools

    def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        return [tool_call("founder_knowledge_retriever", query=query)] if self._has("founder_knowledge_retriever") else []

    # --- step 1: choose tools ------------------------------------------------
    def _plan(self, question: str, mode: str) -> List[Dict[str, Any]]:
        intent = detect_intent(question)
        if intent == "greeting":
            return []
        if intent == "calc":
            expr = extract_expression(question)
            return [tool_call("calculator", expression=expr)] if expr else self._retrieve(question)
        if intent == "daily_plan":
            return [tool_call("daily_planner", goals_and_context=question)] + self._retrieve(
                "daily routine typical workflow morning afternoon evening daily priorities deep work")
        if intent == "project_plan":
            return [tool_call("project_planner", project_idea=question)] + self._retrieve(
                "technology stack frontend backend deployment vector databases automation quick facts") + self._retrieve(
                "product development workflow problem research architecture prototype testing improvement")
        if intent == "resume":
            return [tool_call("resume_coach", query=question)] + self._retrieve("skills strengths experience interview answers")
        if intent == "advisor":
            return [tool_call("founder_advisor", query=question)] + self._retrieve(
                "decision framework questions project selection business thinking")
        if intent == "web":
            return [tool_call("web_search", query=question)]
        if intent == "profile_work":
            return self._retrieve("projects objective technologies lessons learned philosophy") + self._retrieve(
                "technology stack quick facts primary language frontend automation deployment")
        return self._retrieve("introduction professional interests current focus personal mission working style what I enjoy building")

    # --- step 2: answer from tool results -------------------------------------
    def _answer(self, question: str, mode: str, results: List[ToolMessage]) -> str:
        intent = detect_intent(question)
        by_tool: Dict[str, List[str]] = {}
        for r in results:
            by_tool.setdefault(r.name or "", []).append(str(r.content))
        kb_text = "\n\n".join(by_tool.get("founder_knowledge_retriever", []))
        if "unavailable" in kb_text.lower() and len(kb_text) < 200:
            kb_text = ""
        blocks = parse_blocks(kb_text)

        if intent == "greeting":
            return "Hi! I'm FounderOS, Srishanth's digital twin. Ask me about his projects, skills, plans or a business decision."
        if intent == "calc":
            return self._answer_calc(question, by_tool.get("calculator", [""])[0])
        if intent == "daily_plan":
            return self._answer_day(question, blocks)
        if intent == "project_plan":
            return self._answer_project(question, blocks)
        if intent == "web":
            out = by_tool.get("web_search", [""])[0]
            if not out or out.startswith("Error"):
                return "I tried a live web search, but the search service is not reachable from this environment right now, so I can't give you current news. I won't guess."
            return f"Here's what a live web search returned:\n\n{out[:900]}"
        if not blocks:
            return "I don't have that in my knowledge base, so I won't guess."
        if intent == "profile_work":
            return self._answer_work(mode, blocks)
        if intent == "advisor" and find(blocks, "Questions I Ask"):
            return self._answer_advice(blocks)
        if intent in {"resume", "advisor"}:
            return self._answer_generic(question, mode, blocks)
        return self._answer_profile(mode, blocks)

    def _answer_calc(self, question: str, result: str) -> str:
        expr = extract_expression(question) or ""
        try:
            value = float(result)
        except ValueError:
            return f"The calculator couldn't evaluate that ({result}). Could you rephrase the numbers?"
        t = question.lower().replace(",", "")
        nums = [float(n) for n in re.findall(r"\$?\s*(\d+(?:\.\d+)?)", t)]
        if "roi" in t and len(nums) >= 2:
            cost, ret = nums[0], nums[1]
            return (
                f"**ROI = {value:g}%**\n\n"
                f"- Invested: ${cost:,.0f}\n- Returned: ${ret:,.0f}\n- Net gain: ${ret - cost:,.0f}\n\n"
                f"Calculation: `{expr}` = **{value:g}%**, a {ret / cost:g}x return on the money put in.\n\n"
                "My take: that's a strong result. Before scaling it, I'd check whether it is repeatable and how long the payback took."
            )
        return f"`{expr}` = **{value:g}**"

    def _answer_profile(self, mode: str, blocks: List[Dict[str, Any]]) -> str:
        intro = find(blocks, "Tell me about yourself") or find(blocks, "Introduction")
        focus = find(blocks, "Current Focus") or find(blocks, "Professional Interests")
        enjoy = find(blocks, "What I Enjoy Building")
        mission = find(blocks, "Personal Mission")
        style = find(blocks, "Working Style")
        out: List[str] = []
        if mode == "recruiter":
            out.append("Here's a quick profile of Srishanth, from his knowledge base:")
            if intro:
                out.append(f"In his words: “{first_sentences(intro['paras'], 2)}”")
        else:
            name = re.search(r'"name":\s*"([^"]+)"', kb_raw(blocks))
            opener = f"Hi, I'm {name.group(1)}. " if name else ""
            out.append(opener + (first_sentences(intro["paras"], 2) if intro else "Here's a bit about me."))
        if focus:
            out.append(f"**{'Current focus' if focus['heading'] == 'Current Focus' else 'Interests'}:** {join_items(focus['items'])}")
        if enjoy:
            out.append(f"**What {'he enjoys' if mode == 'recruiter' else 'I enjoy'} building:** {join_items(enjoy['items'], 5)}")
        if mission:
            out.append(f"**Mission:** {first_sentences(mission['paras'], 1)}")
        if style:
            steps = " → ".join(i.rstrip(".") for i in style["items"][:5])
            out.append(f"**How {'he works' if mode == 'recruiter' else 'I work'}:** {steps}")
        if mode == "recruiter":
            out.append("Happy to walk you through any of his projects in more detail.")
        return "\n\n".join(out)

    def _answer_work(self, mode: str, blocks: List[Dict[str, Any]]) -> str:
        projects = []
        for b in blocks:
            if b["heading"] == "Objective" and b["top"] and b["top"] not in {p["name"] for p in projects}:
                tech = next((x for x in blocks if x["top"] == b["top"] and x["heading"] == "Technologies"), None)
                objective = first_sentences(b["paras"], 1) if b["paras"] else join_items(b["items"], 4)
                # A chunk boundary can cut a list off from its lead-in, so look at every copy of the block.
                obj_items = next((x["items"] for x in blocks if x["top"] == b["top"] and x["heading"] == "Objective" and x["items"]), [])
                if objective.endswith(":") and obj_items:
                    objective = f"{objective[:-1]} {join_items([i[0].lower() + i[1:] for i in obj_items], 4)}"
                elif objective.endswith(":"):
                    objective = re.sub(r"\s+(capable of|such as|including)\s*:$", "", objective)
                if mode == "recruiter":
                    objective = re.sub(r"\bmyself\b", "himself", objective)
                projects.append({"name": b["top"], "objective": objective.rstrip(":"), "tech": tech["items"] if tech else []})
        stack_lines = [p for b in blocks if b["heading"] == "Quick Facts" for p in b["paras"]
                       if re.match(r"^[A-Z][\w ]+: \S", p)]
        philosophy = find(blocks, "Philosophy")
        who = "Srishanth" if mode == "recruiter" else "I"
        out = [f"Here's what {who} {'has' if mode == 'recruiter' else 'have'} built and the stack behind it:" if projects else "Here's the stack I work with:"]
        if projects:
            out.append("**Projects**")
            out.append("\n".join(
                f"- **{p['name']}**: {p['objective']}" + (f" _({join_items(p['tech'], 5)})_" if p["tech"] else "")
                for p in projects))
        if stack_lines:
            out.append("**Tech stack**")
            out.append("\n".join(f"- {line}" for line in stack_lines[:6]))
        if philosophy and philosophy["paras"]:
            quote = first_sentences(philosophy["paras"], 1)
            out.append(f"In his words: “{quote}”" if mode == "recruiter" else quote)
        if mode == "recruiter":
            out.append("Would you like a deeper walkthrough of any of these projects?")
        return "\n\n".join(out)

    def _answer_day(self, question: str, blocks: List[Dict[str, Any]]) -> str:
        tasks = extract_tasks(question)
        slots = {"Morning": [], "Afternoon": [], "Evening": []}
        for task in tasks:
            t = task.lower()
            if re.search(r"learn|read|study|research|course", t):
                slots["Morning"].append(task)
            elif re.search(r"write|proposal|email|review|document|post|reply", t):
                slots["Evening"].append(task)
            else:
                slots["Afternoon"].append(task)
        routine = {b["label"]: b["items"] for b in blocks if b["heading"] == "Typical Workflow" and b["label"]}
        priorities = find(blocks, "Daily Priorities")
        out = ["Here's your plan for today, built around your usual deep-work routine:"]
        for slot, extra in slots.items():
            habits = routine.get(slot, [])[:2]
            items = [f"**{e[0].upper() + e[1:]}**" for e in extra] + habits
            if items:
                out.append(f"**{slot}**\n" + "\n".join(f"- {i}" for i in items))
        if priorities:
            out.append(f"**Priority order:** {' → '.join(priorities['items'][:5])}")
        out.append("One task at a time, with no multitasking. Tell me if anything moves and I'll re-plan.")
        return "\n\n".join(out)

    def _answer_project(self, question: str, blocks: List[Dict[str, Any]]) -> str:
        def items(heading: str, label: Optional[str] = None, n: int = 3) -> List[str]:
            b = find(blocks, heading, label=label)
            return b["items"][:n] if b else []

        front = items("Frontend", "Preferred Frameworks") or items("Frontend")
        ui = items("Frontend", "Preferred UI", 2)
        backend = items("Backend", "Preferred Technologies", 2) or items("Backend")
        ai = items("Artificial Intelligence", "Primary Interests", 4) or items("Artificial Intelligence")
        vector = items("Vector Databases", "Experience", 1)
        automation = find(blocks, "Automation")
        deploy = items("Deployment", "Preferred Platforms", 2)
        later = items("Deployment", "Interested", 2)
        idea = re.sub(r"^(help me )?plan (a |an )?(new )?", "", question.strip().rstrip("."), flags=re.I)
        arch = []
        if front:
            arch.append(f"- **Frontend:** {' / '.join(front)}" + (f" with {', '.join(ui)}" if ui else ""))
        if backend:
            arch.append(f"- **Backend API:** {' or '.join(backend)}")
        if ai:
            arch.append(f"- **AI layer:** {', '.join(ai)}" + (f", with {vector[0]} as the vector store" if vector else ""))
        platform = [p for p in (automation["paras"] if automation else []) if not p.lower().startswith("favorite")]
        if platform:
            arch.append(f"- **Automations:** {platform[0]} workflows")
        out = [f"Here's how I'd plan **{idea}**:", "**Architecture**", "\n".join(arch) or "- (stack not found in my knowledge base)"]
        workflow = next((b for b in blocks if b["heading"] == "Product Development"), None)
        phases = [p for p in (workflow["paras"] if workflow else []) if len(p.split()) <= 2 and not p.endswith(":")][:6]
        if phases:
            out.append("**Timeline (my usual build loop)**\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(phases, 1)))
        out.append(
            "**Risks**\n- Scope creep: ship a small MVP first\n- LLM cost and latency: cache answers and right-size models\n"
            "- Weak retrieval: test the RAG answers on real questions before launch")
        if deploy:
            out.append(f"**Deployment:** {' and '.join(deploy)} to start" + (f"; {' / '.join(later)} when it needs to scale." if later else "."))
        return "\n\n".join(out)

    def _answer_advice(self, blocks: List[Dict[str, Any]]) -> str:
        questions = find(blocks, "Questions I Ask")["paras"][:6]
        selection = find(blocks, "Project Selection")
        choice = find(blocks, "Technology Choice")
        out = ["I'd run it through the same checklist I use for every big decision:",
               "\n".join(f"- {q}" for q in questions)]
        if selection and selection["items"]:
            out.append(f"**Projects I take on** {join_items(selection['items'])}.")
        if choice and choice["items"]:
            out.append(f"**Technology choices** are judged on {join_items(choice['items'])}, not hype.")
        out.append("If most of those answers are yes, it's worth taking. If not, I'd pass or renegotiate the scope.")
        return "\n\n".join(out)

    def _answer_generic(self, question: str, mode: str, blocks: List[Dict[str, Any]]) -> str:
        q = set(_terms(question))
        scored = sorted(blocks, key=lambda b: -len(q & set(_terms(" ".join([b["heading"], b["label"] or ""] + b["items"] + b["paras"])))))
        out = ["From my knowledge base:" if mode != "recruiter" else "From Srishanth's knowledge base:"]
        for b in scored[:4]:
            title = b["label"] or b["heading"]
            body = join_items(b["items"]) if b["items"] else first_sentences(b["paras"], 2)
            out.append(f"**{title}:** {body}")
        return "\n\n".join(out)


def install() -> None:
    """Register a stand-in `langchain_google_genai` module before FounderOS imports it."""
    shim = types.ModuleType("langchain_google_genai")
    shim.ChatGoogleGenerativeAI = OfflineFounderModel
    shim.GoogleGenerativeAIEmbeddings = LocalHashEmbeddings
    sys.modules["langchain_google_genai"] = shim


if __name__ == "__main__":  # quick manual check: python offline_model.py "question"
    print(json.dumps(detect_intent(" ".join(sys.argv[1:])), indent=2))
