"""Retrieval evaluation: for each question, is the right page in the top-k?

Usage:  python -m eval.eval
Edit eval/questions.json with questions about YOUR uploaded documents.
"""
import json
from pathlib import Path

from app import config
from app.vectorstore import get_vectorstore

QUESTIONS = Path(__file__).with_name("questions.json")


def main():
    items = json.loads(QUESTIONS.read_text())
    vs = get_vectorstore()
    hits = 0
    for it in items:
        results = vs.similarity_search(it["question"], k=config.TOP_K)
        found = any(
            r.metadata["source"] == it["expected_source"] and r.metadata["page"] == it["expected_page"]
            for r in results
        )
        hits += found
        print(("PASS " if found else "FAIL ") + it["question"])
    print(f"\nHit@{config.TOP_K}: {hits}/{len(items)} = {hits / len(items):.0%}")


if __name__ == "__main__":
    main()
