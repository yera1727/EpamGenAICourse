"""Step 4: check the embeddings client.

Embeds a few support-style queries and knowledge snippets and prints the cosine similarity
matrix: each query should be most similar to the snippet that answers it.

Usage:
    python scripts/check_embeddings.py
    python scripts/check_embeddings.py "any text you want to embed"
"""

from __future__ import annotations

import sys
import time

from orbitra_rag.embeddings import EmbeddingsClient, cosine_similarity
from orbitra_rag.providers import ConfigurationError

QUERIES = [
    "My vacuum robot shows E06",
    "How do I remove limescale from the coffee machine?",
    "Thermostat for electric baseboard heaters?",
]
SNIPPETS = [
    "RoboVac R7 error E06: LiDAR sensor blocked. Check that nothing is on top of the LiDAR cover.",
    "BrewMate C3 descaling: pour Orbitra DecalPro into the water tank and start Menu > Maintenance > Descale.",
    "ThermoSmart T2 is not compatible with 120/240 V line-voltage systems such as electric baseboard heaters.",
]


def main() -> int:
    try:
        client = EmbeddingsClient()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 1
    print(f"Provider: {client.settings.provider} | model: {client.model} | expected dim: {client.dim}")

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        vec = client.embed_query(text)
        print(f"'{text}' -> vector of {len(vec)} floats, first 5: {[round(x, 4) for x in vec[:5]]}")
        return 0

    started = time.perf_counter()
    vectors = client.embed_documents(QUERIES + SNIPPETS)  # one batched API call
    elapsed = time.perf_counter() - started
    q_vecs, s_vecs = vectors[: len(QUERIES)], vectors[len(QUERIES) :]
    print(f"Embedded {len(vectors)} texts in {elapsed:.2f}s, vector size = {len(vectors[0])}\n")

    print("Cosine similarity (rows = queries, cols = snippets S1..S3):")
    correct = 0
    for i, (query, qv) in enumerate(zip(QUERIES, q_vecs)):
        sims = [cosine_similarity(qv, sv) for sv in s_vecs]
        best = max(range(len(sims)), key=sims.__getitem__)
        correct += best == i
        cells = "  ".join(f"{'*' if j == best else ' '}{s:.3f}" for j, s in enumerate(sims))
        print(f"  Q{i + 1} {cells}   {query}")
    print(f"\nBest match is the correct snippet for {correct}/{len(QUERIES)} queries (* = best).")
    return 0 if correct == len(QUERIES) else 2


if __name__ == "__main__":
    sys.exit(main())
