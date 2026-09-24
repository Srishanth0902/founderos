from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression. Useful for calculating ROI, Profit, Pricing, Percentages, and Time estimation.
    Input should be a mathematical expression as a string (e.g., '500 * 0.20').
    """
    try:
        # safe evaluation of math expressions
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {e}"
