from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

search_run = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """
    Searches the web using DuckDuckGo. 
    Useful for finding recent AI news, latest frameworks, company information, and technologies.
    """
    try:
        return search_run.run(query)
    except Exception as e:
        return f"Error during web search: {e}"
