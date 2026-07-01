"""
Step 3: LangGraph multi-agent graph.
Router decides: RAG (questions about your docs) vs tool-calling agent
(web search / calculator / general knowledge).
Conversation memory is kept across turns via LangGraph's MemorySaver.
"""
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.llm import get_llm
from src.rag_chain import build_rag_chain
from src.tools import ALL_TOOLS


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    route: str


ROUTER_PROMPT = """You are a router. Given the user's message, decide:
- "rag"   → question that should be answered from the user's uploaded documents
- "agent" → needs live web search, a calculation, or is general conversation

Reply with ONLY one word: rag or agent.

User message: {message}"""


def router_node(state: AgentState):
    llm = get_llm()
    last_message = state["messages"][-1].content
    decision = llm.invoke(ROUTER_PROMPT.format(message=last_message)).content.strip().lower()
    route = "rag" if "rag" in decision else "agent"
    return {"route": route}


def rag_node(state: AgentState):
    chain, _ = build_rag_chain()
    question = state["messages"][-1].content
    answer = chain.invoke(question)
    return {"messages": [SystemMessage(content=answer, name="rag")]}


def agent_node(state: AgentState):
    llm = get_llm().bind_tools(ALL_TOOLS)
    response = llm.invoke(list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"rag": "rag", "agent": "agent"},
    )
    graph.add_edge("rag", END)
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    app = build_graph()
    config = {"configurable": {"thread_id": "test"}}
    result = app.invoke(
        {"messages": [HumanMessage(content="What is 24 * 7?")], "route": ""},
        config=config,
    )
    print(result["messages"][-1].content)
