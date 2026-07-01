"""
Step 1: Ingestion pipeline
Loads PDFs from /data, chunks them, embeds, stores in ChromaDB.

Run: python src/ingest.py
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma                          
from langchain_huggingface import HuggingFaceEmbeddings     

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"


def get_embedding_model():
    # Free, local, CPU-friendly. No API key, no GPU needed.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_and_split_documents():
    all_docs = []
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"No PDFs found in {DATA_DIR}/. Add some PDF files first.")

    for filename in pdf_files:
        path = os.path.join(DATA_DIR, filename)
        print(f"  Loading {filename}...")
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"  {len(all_docs)} pages → {len(chunks)} chunks")
    return chunks


def build_vectorstore():
    print("Building vector store...")
    chunks = load_and_split_documents()
    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"✅ Done — saved to ./{PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
