# Orbitra Support KB — Dataset Description

Synthetic customer-support knowledge base for the fictional brand **Orbitra Home**. Created with LLM
assistance and manually reviewed so that every fact (intervals, codes, limits, versions) is
consistent across all files. Because the products are invented, an LLM cannot answer these questions
from its training data. This makes the effect of retrieval easy to see.

## Contents

```
data/
├── raw/
│   ├── catalog.json                    # manifest: products + list of documents
│   ├── manuals/                        # 4 user manuals (markdown, YAML front matter)
│   │   ├── robovac-r7.md
│   │   ├── brewmate-c3.md
│   │   ├── airpure-a5.md
│   │   └── thermosmart-t2.md
│   ├── troubleshooting/error_codes.csv # 27 error codes across all products
│   ├── faq/faq.jsonl                   # 23 Q&A pairs
│   ├── policies/                       # warranty, returns & support (cross-product)
│   │   ├── warranty.md
│   │   └── returns-and-support.md
│   └── release_notes/firmware.md       # 7 firmware releases
└── eval/eval_questions.jsonl           # 30 evaluation questions (27 answerable + 3 out-of-scope)
```

## Products

| `product_id` | Name | Category | Connected (Wi-Fi) |
|---|---|---|---|
| `robovac-r7` | Orbitra RoboVac R7 | Robot vacuum & mop | yes |
| `brewmate-c3` | Orbitra BrewMate C3 | Bean-to-cup coffee machine | no |
| `airpure-a5` | Orbitra AirPure A5 | Air purifier | yes |
| `thermosmart-t2` | Orbitra ThermoSmart T2 | Smart thermostat | yes |

`product_id = "all"` marks cross-product content (policies, general FAQ).

## Annotation schema

### Markdown documents (`manuals`, `policies`, `release_notes`)

Each file starts with YAML front matter:

| Field | Type | Example | Meaning |
|---|---|---|---|
| `doc_id` | string | `manual-robovac-r7` | stable unique document ID (used in citations) |
| `product_id` | string | `robovac-r7` / `all` | product the document belongs to |
| `doc_type` | enum | `manual`, `policy`, `release_notes` | document type |
| `title` | string | | human-readable title |
| `version` | string | `"2.4"` | document version |
| `firmware` | string | `"2.4.0"` | firmware the manual describes (manuals only) |
| `language` | string | `en` | ISO language code |
| `last_updated` | date | `2026-08-02` | last revision date |

The document body is structured with `##` / `###` headings. **One section is one chunk**, and the
heading becomes the `section` metadata of the chunk.
In `release_notes/firmware.md` every `##` heading has the format
`<product_id> <version> (<YYYY-MM-DD>)`, so each chunk can be annotated with its own `product_id`,
`firmware_version` and `release_date`.

### `troubleshooting/error_codes.csv`

| Column | Meaning |
|---|---|
| `product_id` | product the code belongs to |
| `code` | error code shown by the device (`E01`, `C6`, `F4`, `H3`, …) |
| `title` | short name of the error |
| `severity` | `warning` (device still works), `error` (user can fix it), `critical` (contact support) |
| `cause` | probable cause |
| `solution` | steps to fix it |
| `related_section` | manual section with more detail (links a row to the manual chunk) |

Error code prefixes are unique per product: `E` = RoboVac, `C` = BrewMate, `F` = AirPure, `H` = ThermoSmart.

### `faq/faq.jsonl`

One JSON object per line: `id`, `product_id`, `question`, `answer`, `tags` (list of topic keywords).

### `eval/eval_questions.jsonl`

| Field | Meaning |
|---|---|
| `id` | question ID |
| `question` | user question, phrased the way customers type (sometimes short, like "Coffee machine shows C4") |
| `product_id` | product it relates to (`all` if general) |
| `expected_answer` | reference answer; `OUT_OF_SCOPE: …` means the assistant should refuse or say it does not know |
| `expected_sources` | `doc_id`s that contain the answer (for retrieval hit@k evaluation) |
| `category` | troubleshooting, procedure, specifications, maintenance, warranty, out_of_scope, … |

## Representativeness

* **4 product categories** with different vocabularies (cleaning, coffee, air quality, HVAC).
* **5 document types** with different structure: long prose, tables, rows, Q&A, versioned notes.
* **Some facts appear in more than one document** (e.g. the geofencing radius is in the manual, the
  FAQ and the release notes), as in real knowledge bases.
* **Similar facts for different products** that retrieval must not mix up, for example "reset the
  filter" or "Wi-Fi 2.4 GHz only" appear for several products.
* **Out-of-scope questions** test that the assistant does not hallucinate.

## Size

9 source files. After chunking this gives about 120 chunks: about 60 manual/policy sections,
27 error codes, 23 FAQ pairs and 7 release-note entries.
