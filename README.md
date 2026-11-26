# Orbitra Assist — RAG-based Customer Support Assistant

> Practical task: *Development of RAG-based AI system* (EPAM Gen AI course, module 03).

**Demo video:** _link will be added here after the final step_ — `TODO`

---

## 1. Main idea

**Orbitra Assist** is an AI support assistant for the (fictional) smart-home appliance brand
**Orbitra Home**. A customer asks a question in natural language — *"My vacuum shows E06, what
should I do?"*, *"How often do I descale the coffee machine?"*, *"Is a broken battery covered by
warranty after 18 months?"* — and the assistant answers **strictly based on the official product
documentation**, citing the documents it used.

Why this idea is a good fit for RAG:

* **The knowledge is private and non-public.** Orbitra Home is invented, so no LLM has its manuals,
  error codes or policies in its training data. Without retrieval the model can only hallucinate;
  with retrieval it answers correctly. This makes the value of RAG directly measurable.
* **Answers must be precise.** Error codes, intervals ("replace every 3 months"), limits
  ("2.4 GHz Wi-Fi only", "max 20 mm thresholds") and policy terms are exact facts — a typical
  real-world support scenario where hallucinations are expensive.
* **The data is heterogeneous.** Long-form manuals, tabular error codes, Q&A pairs, legal-style
  policies and versioned release notes require different chunking strategies and metadata, which
  makes the retrieval part non-trivial.

## 2. Key concepts

| Concept | How it is used in the project |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | The user question is embedded, the most similar knowledge chunks are retrieved from a vector DB and placed into the LLM context together with the question. |
| **Embeddings** | Every chunk of the dataset and every user query is converted into a dense vector by the same embedding model, so semantic similarity = vector proximity. |
| **Vector search** | Cosine similarity / ANN (HNSW index) search over chunk vectors, top-*k* results. |
| **Metadata filtering** | Each chunk carries a payload (`product_id`, `doc_type`, `section`, ...). Search can be narrowed to one product (e.g. only *BrewMate C3* docs). |
| **Grounded generation** | The system prompt instructs the LLM to answer only from the provided context, cite sources, and say "I don't know" when the context has no answer. |
| **Evaluation set** | A hand-written set of questions with expected answers and expected source documents, used to check retrieval quality (hit@k) and answer correctness. |

## 3. Dataset concept

The dataset is a **synthetic but realistic product knowledge base** for four Orbitra Home products,
created with LLM assistance and then manually reviewed for internal consistency (the same facts —
intervals, codes, limits — match across manuals, FAQ, error tables and the evaluation set).

| Product ID | Product | Category |
|---|---|---|
| `robovac-r7` | Orbitra RoboVac R7 | Robot vacuum & mop |
| `brewmate-c3` | Orbitra BrewMate C3 | Bean-to-cup coffee machine |
| `airpure-a5` | Orbitra AirPure A5 | Air purifier |
| `thermosmart-t2` | Orbitra ThermoSmart T2 | Smart thermostat |

### Document types

| `doc_type` | Files | Format | Chunking unit |
|---|---|---|---|
| `manual` | `data/raw/manuals/*.md` | Markdown + YAML front matter | Section (`##` / `###` heading) |
| `troubleshooting` | `data/raw/troubleshooting/error_codes.csv` | CSV, one row per error code | One row = one chunk |
| `faq` | `data/raw/faq/faq.jsonl` | JSON Lines, one Q&A per line | One Q&A pair = one chunk |
| `policy` | `data/raw/policies/*.md` | Markdown + YAML front matter | Section |
| `release_notes` | `data/raw/release_notes/firmware.md` | Markdown + YAML front matter | One firmware version = one chunk |

Supporting files:

* `data/raw/catalog.json` — the dataset manifest: products and the list of all source documents
  with their metadata.
* `data/eval/eval_questions.jsonl` — evaluation questions with expected answers and the IDs of the
  documents that contain the answer (includes "out-of-scope" questions to test refusal behaviour).
* `data/README.md` — detailed description of the dataset schema and annotation rules.

### Annotation

Every source unit is annotated so that each future chunk can carry rich metadata:

* `doc_id` — stable unique ID of the source document (e.g. `manual-robovac-r7`).
* `product_id` — product the content belongs to, or `all` for cross-product docs (policies).
* `doc_type` — one of the types above.
* `title`, `version`, `last_updated`, `language`.
* Additionally per type: `section` (manuals/policies), `code` + `severity` (error codes),
  `tags` (FAQ), `firmware_version` + `release_date` (release notes).

## 4. Design

### Architecture

```
                 ┌──────────────────────── offline: ingestion ────────────────────────┐
 data/raw/*  ──► │ loaders ─► chunker (per doc_type) ─► embeddings client ─► upsert   │ ──► Vector DB
                 └────────────────────────────────────────────────────────────────────┘     (Qdrant)
                                                                                              ▲
                 ┌──────────────────────── online: question answering ─────────────────┐     │
 User ─► UI ───► │ query ─► embeddings client ─► vector search (top-k, filters) ───────┼─────┘
   ▲             │                                   │                                 │
   │             │            prompt = system rules + retrieved chunks + question     │
   │             │                                   ▼                                 │
   └──── answer + sources ◄──────────────── LLM client (DIAL) ◄────────────────────────┘
                 └─────────────────────────────────────────────────────────────────────┘
```

### Components

1. **Data loaders & chunker** — read every source format, split it into semantically complete
   chunks (sections, rows, Q&A pairs, versions), prepend a short context header
   (product + document title + section) to each chunk text to improve embedding quality, and
   attach metadata.
2. **Embeddings client** — wraps the embedding model API, supports batching and retries; the same
   client is used for ingestion and for queries.
3. **Vector database** — stores vectors + payload (chunk text and metadata), cosine distance,
   HNSW index, payload index on `product_id` and `doc_type` for filtering.
4. **Ingestion script** — one command that (re)creates the collection, chunks the dataset, embeds
   it and uploads it; idempotent (deterministic chunk IDs, so re-running does not duplicate data).
5. **LLM client** — wraps the chat-completion API of EPAM DIAL (OpenAI-compatible), handles simple
   request/response and the RAG prompt.
6. **RAG pipeline** — query → embedding → top-*k* search (optional product filter, score
   threshold) → prompt assembly → LLM → answer with cited sources.
7. **UI** — Streamlit web app: chat window, product filter, top-*k* slider, an expandable panel
   showing the retrieved chunks with similarity scores, and a "compare with no-RAG answer" toggle
   to demonstrate the effect of retrieval.

### Prompting rules (grounding)

* Answer **only** using the provided context; if the context is insufficient, say so and suggest
  contacting Orbitra support.
* Cite sources as `[doc_id › section]`.
* Keep answers short and actionable (numbered steps for procedures).

## 5. System technical details

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Vector DB | **Qdrant** (Docker image `qdrant/qdrant`) | Free, local, native vector search + payload filtering, web dashboard on `:6333/dashboard`. |
| Embeddings | `text-embedding-3-small` via DIAL (OpenAI-compatible API) | 1536-dim vectors, cosine distance. Model name and dimension are configurable. |
| LLM | `gpt-4o` (or other DIAL deployment) | Configurable deployment name; temperature ≈ 0.1 for factual answers. |
| UI | Streamlit | Plus a minimal CLI for quick testing. |
| Config | `.env` file (`DIAL_API_KEY`, `DIAL_ENDPOINT`, model names, Qdrant URL) | Secrets are never committed (`.env.example` only). |
| Infrastructure | `docker-compose.yml` for Qdrant | Persistent volume for DB storage. |

Planned repository layout:

```
.
├── README.md                  # this file
├── data/
│   ├── README.md              # dataset schema & annotation rules
│   ├── raw/                   # source dataset (manuals, error codes, FAQ, policies, release notes)
│   └── eval/                  # evaluation questions
├── docker-compose.yml         # Qdrant
├── src/orbitra_rag/           # chunking, embeddings client, vector store, LLM client, RAG pipeline
├── scripts/ingest.py          # fill the DB
├── app.py                     # Streamlit UI
├── requirements.txt
└── .env.example
```

## 6. Requirements

### Functional

* FR-1. The user can ask a free-text question about any Orbitra product through the UI.
* FR-2. The question is converted to an embedding and used for vector search in the DB.
* FR-3. The top-*k* most relevant chunks (optionally filtered by product) are passed to the LLM
  together with the question in one context window.
* FR-4. The answer is displayed with the list of sources (document, section, similarity score).
* FR-5. If no relevant information is found, the assistant explicitly says so instead of guessing.
* FR-6. The whole dataset can be (re)loaded into the DB with a single script.

### Non-functional

* NFR-1. Runs locally: DB in Docker, application in a Python virtual environment.
* NFR-2. Response time for a typical question ≤ ~5 s (dominated by the LLM call).
* NFR-3. Secrets are stored only in environment variables / `.env`.
* NFR-4. Ingestion is idempotent and reproducible.

## 7. Limitations

* **Synthetic data.** The dataset describes fictional products; it is small (9 source files,
  ~120 chunks) but representative of a real support knowledge base.
* **English only.** Documents are in English; questions in other languages may still work thanks to
  multilingual embeddings, but retrieval quality is not guaranteed.
* **Single-turn retrieval.** Each question is retrieved independently; follow-up questions that rely
  on chat history ("and what about the other one?") may retrieve poorly.
* **No re-ranking / hybrid search** in the baseline version (pure dense retrieval); exact-code queries
  like "E06" rely on the context header included in each chunk.
* **Context window budget.** Only top-*k* chunks (default 5) are sent to the LLM, so questions that
  need information spread across many documents may get incomplete answers.
* **External dependency.** Embedding and LLM calls require network access to DIAL and a valid API key.
* Not a production system: no authentication, no conversation storage, no monitoring.
