"""Evaluate retrieval quality on data/eval/eval_questions.jsonl.

For every answerable question we check whether at least one of its `expected_sources` (doc_ids)
appears among the top-k retrieved chunks (hit@k), and compute MRR of the first relevant chunk.
Out-of-scope questions are reported separately with their best similarity score, which helps to
choose a score threshold for "I don't know" answers.

Usage:
    python scripts/eval_retrieval.py            # no product filter (harder)
    python scripts/eval_retrieval.py --filter   # use each question's product_id as a filter
"""

from __future__ import annotations

import argparse
import json
import sys

from orbitra_rag.config import DATA_DIR
from orbitra_rag.embeddings import EmbeddingsClient
from orbitra_rag.vector_store import VectorStore

KS = (1, 3, 5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", action="store_true", help="filter search by the question's product_id")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every question")
    args = parser.parse_args()

    questions = [json.loads(line) for line in (DATA_DIR / "eval" / "eval_questions.jsonl").read_text("utf-8").splitlines() if line.strip()]
    embedder, store = EmbeddingsClient(), VectorStore()
    vectors = embedder.embed_documents([q["question"] for q in questions])

    hits = {k: 0 for k in KS}
    reciprocal_ranks, answerable, misses, oos = [], 0, [], []
    for q, vec in zip(questions, vectors):
        product = q["product_id"] if args.filter and q["product_id"] != "all" else None
        results = store.search(vec, top_k=max(KS), product_id=product)
        if not q["expected_sources"]:
            oos.append((q["question"], results[0].score if results else 0.0))
            continue

        answerable += 1
        expected = set(q["expected_sources"])
        rank = next((i + 1 for i, r in enumerate(results) if r.payload["doc_id"] in expected), None)
        for k in KS:
            hits[k] += rank is not None and rank <= k
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if rank != 1:
            misses.append((q["id"], q["question"], rank, [r.payload["chunk_id"] for r in results[:3]]))
        if args.verbose:
            print(f"{q['id']} rank={rank}  top1={results[0].payload['chunk_id']}  {q['question']}")

    mode = "with product filter" if args.filter else "without filter"
    print(f"\nRetrieval evaluation ({answerable} answerable questions, {mode})")
    for k in KS:
        print(f"  hit@{k}: {hits[k] / answerable:.2%}")
    print(f"  MRR@{max(KS)}: {sum(reciprocal_ranks) / answerable:.3f}")

    if misses:
        print("\nQuestions where the first result is not from an expected document:")
        for qid, question, rank, top in misses:
            print(f"  {qid} (rank {rank}) {question}\n      top-3: {top}")

    print("\nOut-of-scope questions (best similarity score):")
    for question, score in oos:
        print(f"  {score:.3f}  {question}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
