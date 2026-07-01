"""
Tools the agent can call when the answer isn't in the documents.
Uses DuckDuckGo (no API key needed) instead of Tavily.
"""
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for current information not found in the local documents."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"{r['title']}: {r['body']}" for r in results
        )
    except Exception as e:
        return f"Web search failed: {e}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '12 * (4 + 3)'."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "Invalid expression — only numbers and + - * / ( ) allowed."
        return str(eval(expression))
    except Exception as e:
        return f"Calculation failed: {e}"


ALL_TOOLS = [web_search, calculator]
