from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import call_model, tool_node, should_continue
from agent.memory import get_checkpointer

def create_agent_graph():
    """
    Creates the LangGraph agent for FounderOS.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    # Conditional edge from agent to tools or END
    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )
    
    # Edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile graph with memory checkpointer
    checkpointer = get_checkpointer()
    graph = workflow.compile(checkpointer=checkpointer)
    
    return graph
