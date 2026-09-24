from langchain.tools import tool

@tool
def resume_coach(query: str) -> str:
    """
    Acts as a Resume Coach. 
    Use this when the user asks for feedback on their resume, interview questions, strengths/weaknesses, or STAR stories.
    Input should be what the user is asking for (e.g., 'What are my strengths?').
    """
    return (
        f"As a Resume Coach, please review this request: {query}\n"
        "Return Strengths, Weaknesses, STAR Stories, Interview Questions, and Improvements based on my knowledge base."
    )
