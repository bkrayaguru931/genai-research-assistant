"""
Step 4: Streamlit UI.
Run: streamlit run app.py
"""
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent_graph import build_graph
from src.rag_chain import build_rag_chain

st.set_page_config(page_title="Agentic Research Assistant", page_icon="🔎")
st.title("🔎 Agentic Research Assistant")
st.caption("RAG + LangGraph + Tool Calling — ask about your documents or anything else")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
    st.session_state.config = {"configurable": {"thread_id": "streamlit-session"}}
    st.session_state.history = []

# Render past messages
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("Ask a question about your documents, or anything else...")

if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Streaming directly from the RAG chain for the doc-grounded path
        chain, _ = build_rag_chain()
        try:
            for chunk in chain.stream(user_input):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Error: {e}"
            placeholder.markdown(full_response)

    st.session_state.history.append(AIMessage(content=full_response))

with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
1. Run `python src/ingest.py` first to embed your PDFs into ChromaDB
2. This UI streams answers grounded in your documents
3. Swap to `src/agent_graph.py`'s `build_graph()` for the full
   router + tool-calling agent (web search / calculator / general chat)
        """
    )
