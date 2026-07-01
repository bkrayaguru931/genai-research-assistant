# Agentic Research Assistant (RAG + LangGraph + Tool Calling)

<img width="1413" height="794" alt="Screenshot 2026-07-01 at 16 24 06" src="https://github.com/user-attachments/assets/fa3a62b8-cce9-4477-9a82-ce8ac67b2c23" />

An end-to-end GenAI app: upload PDFs, ask questions grounded in them, and
fall back to web search / a calculator / general chat when the docs don't
have the answer — all routed automatically by a LangGraph agent, with
streaming responses and a basic evaluation harness.

**Runs entirely on CPU.** No GPU needed — the LLM call goes to an API
(OpenAI or Gemini), and the embedding model (`all-MiniLM-L6-v2`) is small
enough to run locally on any laptop.

## Tech stack
LLM (OpenAI / Gemini) · LangChain · LangGraph · ChromaDB · Hugging Face
sentence-transformers (embeddings) · Semantic search · Prompt engineering ·
Tool/function calling · Multi-agent routing · Conversation memory ·
Streaming responses (Streamlit) · Retrieval + answer evaluation

## Architecture
```
PDF(s) -> chunk -> embed (HF MiniLM) -> ChromaDB
                                            |
User question -> LangGraph router -----> RAG path  -> grounded, cited answer
                       |
                       -----> Tool-calling agent -> web search / calculator / chat
                                            |
                                  (conversation memory persists across turns)
```

## Setup
```bash
git clone <your-repo-url>
cd genai-research-assistant
python -m venv venv && source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # fill in your API key(s)
```

Get a free Gemini API key at https://aistudio.google.com/apikey (recommended,
no cost) or use OpenAI if you prefer.

## Usage
1. Drop a few PDFs into `data/`
2. Build the vector store: `python src/ingest.py`
3. Launch the app: `streamlit run app.py`
4. (Optional) Edit `evaluation/test_cases.json` with real questions about
   your PDFs, then run `python evaluation/evaluate.py`

## What each file does
- `src/ingest.py` — loads PDFs, chunks them, embeds, stores in ChromaDB
- `src/rag_chain.py` — retriever + grounded prompt + LLM (hallucination
  reduction via "only answer from context, cite sources, say I don't know")
- `src/tools.py` — web search (Tavily) and calculator tools
- `src/agent_graph.py` — LangGraph state machine: router -> RAG or
  tool-calling agent -> tools -> back to agent, with memory checkpointing
- `app.py` — Streamlit chat UI with streaming
- `evaluation/evaluate.py` — checks retrieval hit rate + LLM-as-judge
  scoring of answers against a reference set

## Notes on design choices
- Used a free local Hugging Face embedding model instead of an API
  embedding model to keep cost at zero and avoid GPU dependency.
- ChromaDB chosen over FAISS for simplicity (no manual index management);
  FAISS would be a reasonable swap for larger-scale or production use.
- Hallucination reduction is handled via prompt constraints rather than a
  separate fact-checking model, to keep the pipeline simple — a natural
  next step would be adding a self-critique/verification node in the graph.
