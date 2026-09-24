from langchain.tools import tool

@tool
def founder_advisor(query: str) -> str:
    """
    Analyzes business scenarios (clients, timelines, profits, markets) to provide recommendations.
    Use this when asking for business advice, pricing, or strategic decisions.
    Input should be a detailed scenario description.
    """
    # In a full production system, this could call a separate LLM chain or agent to reason over the input.
    # For now, it returns a structured prompt that the main agent will fulfill using its system prompt.
    return (
        f"Analyze this business scenario as FounderOS: {query}\n"
        "Please provide a recommendation considering Client, Timeline, Profit, and Market."
    )
