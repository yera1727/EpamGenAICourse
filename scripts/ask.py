"""Ask the RAG system from the terminal (CLI alternative to the Streamlit UI).

Usage:
    python scripts/ask.py "My vacuum shows E06"
    python scripts/ask.py "How long does descaling take?" --product brewmate-c3 --top-k 3
    python scripts/ask.py "My vacuum shows E06" --compare     # also show the answer without RAG
"""

from __future__ import annotations

import argparse
import sys

from orbitra_rag.providers import ConfigurationError
from orbitra_rag.rag import RAGPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--product", choices=["robovac-r7", "brewmate-c3", "airpure-a5", "thermosmart-t2"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--compare", action="store_true", help="also ask the LLM without retrieved context")
    args = parser.parse_args()

    try:
        rag = RAGPipeline(top_k=args.top_k)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 1

    result = rag.answer(args.question, product_id=args.product)
    print("Retrieved context:")
    for i, hit in enumerate(result.hits, start=1):
        print(f"  [{i}] {hit.score:.3f}  {hit.payload['doc_id']} › {hit.payload['section']}")
    print(f"\nAnswer (RAG):\n{result.answer}")
    t = result.timings
    print(f"\n[embedding {t['embedding_s']:.2f}s | search {t['search_s'] * 1000:.0f}ms | LLM {t['llm_s']:.2f}s]")

    if args.compare:
        baseline = rag.answer_without_rag(args.question)
        print(f"\nAnswer WITHOUT RAG (same LLM, no context):\n{baseline.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
