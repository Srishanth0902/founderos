import operator
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    State of the FounderOS Agent graph.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    mode: str # e.g., 'normal', 'recruiter', 'daily_assistant'
