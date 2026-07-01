"""
Step 2: RAG chain.
Retriever -> grounded prompt -> LLM.
Prompt forces the model to only use retrieved context and say when it
doesn't know — this is the hallucination-reduction piece.
"""
from langchain_chroma import Chroma                          
from langchain_huggingface import HuggingFaceEmbeddings     
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.llm import get_llm

PERSIST_DIR = "chroma_db"

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert research assistant.

Answer ONLY from the retrieved context.

If the retrieved context contains partial information,
answer using that information.

Only say "I couldn't find this in the provided documents"
if none of the retrieved chunks contain relevant information.

Always mention the source numbers you used.

Context:
{context}

Question:
{question}

Answer:
""")


def get_retriever(k=8):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        },
    )


def format_docs(docs):
    formatted = []

    for i, d in enumerate(docs):
        source = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "?")

        formatted.append(
            f"""
========================
Source {i+1}

File: {source}

Page: {page}

Content:

{d.page_content}
========================
"""
        )

    return "\n".join(formatted)

def retrieve(question):
    docs = retriever.invoke(question)

    print("\n" + "=" * 100)
    print("QUESTION:")
    print(question)
    print("=" * 100)

    for i, d in enumerate(docs):
        print(f"\nChunk {i+1}")
        print("Metadata:", d.metadata)
        print("-" * 60)
        print(d.page_content[:800])
        print()

    return format_docs(docs)

def build_rag_chain():
    global retriever

    retriever = get_retriever()
    llm = get_llm()

    chain = (
        {
            "context": retrieve,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever

if __name__ == "__main__":
    chain, _ = build_rag_chain()
    print(chain.invoke("What is this document about?"))
