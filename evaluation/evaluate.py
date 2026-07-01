"""
Step 5: Evaluation.
Checks (1) retrieval quality - does the retrieved context actually contain
the answer, and (2) answer quality - LLM-as-judge scoring vs expected answer.

Edit test_cases.json with real questions about YOUR documents, then run:
python evaluation/evaluate.py
"""
import json
import os
from src.rag_chain import build_rag_chain, get_retriever
from src.llm import get_llm

JUDGE_PROMPT = """You are grading an AI assistant's answer.

Question: {question}
Expected (reference) answer: {expected}
Model's actual answer: {actual}

Score the model's answer from 1-5 on factual correctness relative to the
expected answer (5 = fully correct, 1 = wrong or hallucinated).
Reply with ONLY the number."""


def load_test_cases():
    path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    with open(path) as f:
        return json.load(f)


def run_evaluation():
    chain, _ = build_rag_chain()
    retriever = get_retriever()
    judge = get_llm()

    test_cases = load_test_cases()
    scores = []

    for case in test_cases:
        question = case["question"]
        expected = case["expected_answer"]

        retrieved_docs = retriever.invoke(question)
        retrieved_text = " ".join(d.page_content.lower() for d in retrieved_docs)
        retrieval_hit = any(
            keyword.lower() in retrieved_text for keyword in case.get("expected_keywords", [])
        )

        actual_answer = chain.invoke(question)

        judge_response = judge.invoke(
            JUDGE_PROMPT.format(question=question, expected=expected, actual=actual_answer)
        ).content.strip()
        try:
            score = int(judge_response[0])
        except (ValueError, IndexError):
            score = 0

        scores.append(score)
        print(f"\nQ: {question}")
        print(f"Retrieval hit: {retrieval_hit}")
        print(f"Answer: {actual_answer[:200]}...")
        print(f"Score: {score}/5")

    avg = sum(scores) / len(scores) if scores else 0
    print(f"\n=== Average score: {avg:.2f}/5 across {len(scores)} questions ===")


if __name__ == "__main__":
    run_evaluation()
