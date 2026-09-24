from langchain.tools import tool

@tool
def project_planner(project_idea: str) -> str:
    """
    Acts as a Project Planner.
    Use this when the user asks to plan a project (e.g., 'Build an AI SaaS').
    Returns Architecture, Timeline, Tech Stack, Risks, and Deployment strategy.
    Input should be the project description.
    """
    return (
        f"Plan this project: {project_idea}\n"
        "Provide Architecture, Timeline, Tech Stack, Risks, and Deployment."
    )

@tool
def daily_planner(goals_and_context: str) -> str:
    """
    Acts as a Daily Planner.
    Use this when the user asks to plan their day.
    Returns Morning, Afternoon, Evening schedule and Priorities.
    Input can be current context or just a request to plan the day.
    """
    return (
        f"Plan my day considering this context: {goals_and_context}\n"
        "Provide Morning, Afternoon, Evening schedules, and Priorities."
    )
