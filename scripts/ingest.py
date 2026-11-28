"""Step 5: fill the vector database with the dataset.

Pipeline: data/raw/* -> chunks (with metadata) -> embeddings -> Qdrant.
The chunks are also written to data/processed/chunks.jsonl for inspection.

Usage:
    python scripts/ingest.py             # rebuild the collection from scratch
    python scripts/ingest.py --dry-run   # only chunk the data and print statistics (no API calls)
    python scripts/ingest.py --append    # upsert into the existing collection (same chunk -> same id)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

from orbitra_rag.chunking import load_chunks
from orbitra_rag.config import DATA_DIR, get_settings
from orbitra_rag.embeddings import EmbeddingsClient
from orbitra_rag.providers import ConfigurationError
from orbitra_rag.vector_store import VectorStore

PROCESSED = DATA_DIR / "processed" / "chunks.jsonl"
SAMPLE_QUERIES = [
    ("robot vacuum shows error E06", None),
    ("how long does descaling take", "brewmate-c3"),
]


def print_stats(chunks) -> None:
    lengths = [len(c.text) for c in chunks]
    print(f"Chunks: {len(chunks)}  (chars: min {min(lengths)}, avg {sum(lengths) // len(lengths)}, max {max(lengths)})")
    print("  by doc_type: ", dict(Counter(c.metadata["doc_type"] for c in chunks)))
    print("  by product:  ", dict(Counter(c.metadata["product_id"] for c in chunks)))


def save_chunks(chunks) -> None:
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c.chunk_id, **c.payload()}, ensure_ascii=False) + "\n")
    print(f"Chunks saved to {PROCESSED.relative_to(DATA_DIR.parent).as_posix()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="chunk only, no embeddings / DB writes")
    parser.add_argument("--append", action="store_true", help="do not recreate the collection")
    args = parser.parse_args()

    chunks = load_chunks()
    print_stats(chunks)
    save_chunks(chunks)
    if args.dry_run:
        print("\nExample chunk:\n" + "-" * 60 + f"\n{chunks[0].text}\n" + "-" * 60)
        return 0

    settings = get_settings()
    try:
        embedder = EmbeddingsClient(settings)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 1
    store = VectorStore(settings)
    store.ensure_collection(recreate=not args.append)

    print(f"\nEmbedding {len(chunks)} chunks with '{embedder.model}' ({embedder.dim} dims)...")
    started = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks])
    print(f"  done in {time.perf_counter() - started:.1f}s")

    store.upsert([c.chunk_id for c in chunks], vectors, [c.payload() for c in chunks])
    info = store.info()
    print(f"Uploaded. Collection '{info['collection']}': {info['points']} points, status {info['status']}")

    print("\nSanity check (vector search):")
    for query, product in SAMPLE_QUERIES:
        hits = store.search(embedder.embed_query(query), top_k=3, product_id=product)
        label = f" [filter: {product}]" if product else ""
        print(f"  '{query}'{label}")
        for h in hits:
            print(f"     {h.score:.3f}  {h.payload['chunk_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
