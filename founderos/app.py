import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import create_agent_graph


def normalize_text_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item:
                    parts.append(normalize_text_content(item["content"]))
                else:
                    parts.append(str(item))
            else:
                text = getattr(item, "text", None)
                if text is None:
                    text = getattr(item, "content", None)
                if text is None:
                    text = str(item)
                parts.append(normalize_text_content(text))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        for key in ("text", "content"):
            if key in content:
                return normalize_text_content(content[key])
        return str(content)

    text = getattr(content, "text", None)
    if text is None:
        text = getattr(content, "content", None)
    if text is not None:
        return normalize_text_content(text)
    return str(content)


# Configure page
st.set_page_config(page_title="FounderOS - My Digital Twin", page_icon="🧠", layout="centered")

st.title("FounderOS – My Digital Twin")
st.markdown("*An AI-powered digital twin that understands my experience, remembers my projects, reasons like me, and helps solve real-world problems.*")

# Initialize graph and session state
@st.cache_resource
def get_graph():
    return create_agent_graph()

graph = get_graph()

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Mode Selection
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox(
        "Select Mode",
        ["Normal Mode", "Recruiter Mode", "Daily Assistant Mode"]
    )
    st.markdown("---")
    st.markdown("### Suggested Actions")
    if st.button("Learn about me"):
        st.session_state.messages.append({"role": "user", "content": "Tell me about yourself."})
    if st.button("Calculate ROI"):
        st.session_state.messages.append({"role": "user", "content": "Calculate the ROI of a $500 investment that yields $1500."})
    if st.button("Plan a project"):
        st.session_state.messages.append({"role": "user", "content": "Help me plan a new AI SaaS project."})

# Map display mode to internal mode
mode_mapping = {
    "Normal Mode": "normal",
    "Recruiter Mode": "recruiter",
    "Daily Assistant Mode": "daily_assistant"
}
internal_mode = mode_mapping[mode]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("FounderOS is thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Format history for LangGraph
            langchain_msgs = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    langchain_msgs.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    langchain_msgs.append(AIMessage(content=m["content"]))
            
            state = {
                "messages": langchain_msgs,
                "mode": internal_mode
            }
            
            response = graph.invoke(state, config=config)

            raw_message = response["messages"][-1]
            final_msg = normalize_text_content(getattr(raw_message, "content", raw_message))
            st.write(final_msg)
            st.session_state.messages.append({"role": "assistant", "content": final_msg})
