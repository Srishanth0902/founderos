from typing import Literal
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
import os
from config import LLM_PROVIDER, CHAT_MODEL, PROMPTS_DIR
from rag.retriever import get_retriever_tool
from tools.calculator import calculator
from tools.search import web_search
from tools.founder_advisor import founder_advisor
from tools.recruiter import resume_coach
from tools.planner import project_planner, daily_planner

# Initialize Tools
try:
    retriever_tool = get_retriever_tool()
    tools = [retriever_tool, calculator, web_search, founder_advisor, resume_coach, project_planner, daily_planner]
except Exception as e:
    print(f"Warning: Could not initialize retriever tool (Vector store missing?). Defaulting to other tools. {e}")
    tools = [calculator, web_search, founder_advisor, resume_coach, project_planner, daily_planner]

def get_llm():
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.7)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=CHAT_MODEL, temperature=0.7)

llm = get_llm()
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

def call_model(state):
    messages = state["messages"]
    
    # Read appropriate system prompt based on mode
    mode = state.get("mode", "normal")
    if mode == "recruiter":
        prompt_file = os.path.join(PROMPTS_DIR, "recruiter_prompt.txt")
    elif mode == "daily_assistant":
        prompt_file = os.path.join(PROMPTS_DIR, "daily_assistant_prompt.txt")
    else:
        prompt_file = os.path.join(PROMPTS_DIR, "system_prompt.txt")
        
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            sys_prompt_text = f.read()
    except:
        sys_prompt_text = "You are FounderOS."
        
    # Prepend system message
    from langchain_core.messages import SystemMessage
    sys_msg = SystemMessage(content=sys_prompt_text)
    
    response = llm_with_tools.invoke([sys_msg] + messages)
    return {"messages": [response]}

def should_continue(state) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"
