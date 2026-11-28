"""Load the raw dataset and split it into annotated, self-contained chunks.

Chunking strategy per document type (see data/README.md):
    manual / policy  -> one chunk per `##` / `###` section
    release_notes    -> one chunk per firmware release (`## <product_id> <version> (<date>)`)
    troubleshooting  -> one chunk per error-code row of the CSV
    faq              -> one chunk per Q&A pair

Every chunk text starts with a short context header (product, document, section). The header is
embedded together with the body, so a section such as "Maintenance" is still associated with the
right product, and it gives the LLM a readable source label.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
MAX_CHUNK_CHARS = 2500  # sections longer than this are split by paragraphs

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
_RELEASE_RE = re.compile(r"^(?P<product>[a-z0-9-]+)\s+(?P<version>\d+\.\d+\.\d+)\s+\((?P<date>\d{4}-\d{2}-\d{2})\)$")


@dataclass
class Chunk:
    chunk_id: str
    text: str  # header + body: what gets embedded and shown to the LLM
    metadata: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {"text": self.text, **self.metadata}


# ---------------------------------------------------------------------------- helpers


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_catalog(raw_dir: Path = RAW_DIR) -> dict[str, Any]:
    return json.loads((raw_dir / "catalog.json").read_text(encoding="utf-8"))


def product_names(catalog: dict[str, Any]) -> dict[str, str]:
    names = {p["product_id"]: p["name"] for p in catalog["products"]}
    names["all"] = "All Orbitra Home products"
    return names


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Parse a simple `key: value` YAML front matter block. Returns (metadata, body)."""
    if not text.startswith("---"):
        return {}, text
    _, block, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in block.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown into (section_path, content) pairs on `##` and `###` headings.

    A `###` section gets the path "Parent › Child". Content before the first `##`
    (the `#` title) is ignored. Sections without content are skipped.
    """
    sections: list[tuple[str, list[str]]] = []
    parent = ""
    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            level, title = match.groups()
            if level == "##":
                parent = title
                sections.append((title, []))
            else:
                sections.append((f"{parent} › {title}" if parent else title, []))
        elif sections:
            sections[-1][1].append(line)
    result = []
    for name, lines in sections:
        content = "\n".join(lines).strip()
        if content:
            result.append((name, content))
    return result


def split_long(content: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split an over-long section into paragraph-aligned parts (no-op for normal sections)."""
    if len(content) <= limit:
        return [content]
    parts, current = [], ""
    for paragraph in content.split("\n\n"):
        # `current` already ends with the "\n\n" separator
        if current and len(current) + len(paragraph) > limit:
            parts.append(current.strip())
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def make_header(product: str, document: str, section: str) -> str:
    return f"Product: {product}\nDocument: {document}\nSection: {section}"


# ---------------------------------------------------------------------------- loaders


def chunk_markdown(path: Path, names: dict[str, str]) -> list[Chunk]:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    doc_id, doc_type = meta["doc_id"], meta["doc_type"]
    base = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "title": meta.get("title", path.stem),
        "version": meta.get("version", ""),
        "last_updated": meta.get("last_updated", ""),
        "source": path.relative_to(RAW_DIR.parent.parent).as_posix(),
    }

    chunks = []
    for section, content in split_sections(body):
        product_id = meta["product_id"]
        extra: dict[str, Any] = {}
        if doc_type == "release_notes":
            release = _RELEASE_RE.match(section)
            if not release:
                raise ValueError(f"{path.name}: release heading '{section}' does not match the expected format")
            product_id = release["product"]
            extra = {"firmware_version": release["version"], "release_date": release["date"]}

        parts = split_long(content)
        for i, part in enumerate(parts):
            suffix = f"-{i + 1}" if len(parts) > 1 else ""
            header = make_header(names.get(product_id, product_id), base["title"], section)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{slugify(section)}{suffix}",
                    text=f"{header}\n\n{part}",
                    metadata={**base, **extra, "product_id": product_id, "section": section},
                )
            )
    return chunks


def chunk_error_codes(path: Path, names: dict[str, str]) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            product = names[row["product_id"]]
            section = f"Error {row['code']}"
            body = (
                f"Error code {row['code']}: {row['title']} ({product})\n"
                f"Severity: {row['severity']}\n"
                f"Cause: {row['cause']}\n"
                f"Solution: {row['solution']}\n"
                f"More details in the user manual, section: {row['related_section']}"
            )
            chunks.append(
                Chunk(
                    chunk_id=f"error-codes#{row['product_id']}-{row['code'].lower()}",
                    text=f"{make_header(product, 'Error codes and troubleshooting', section)}\n\n{body}",
                    metadata={
                        "doc_id": "error-codes",
                        "doc_type": "troubleshooting",
                        "title": "Error codes and troubleshooting",
                        "product_id": row["product_id"],
                        "section": section,
                        "code": row["code"],
                        "severity": row["severity"],
                        "source": "data/raw/troubleshooting/error_codes.csv",
                    },
                )
            )
    return chunks


def chunk_faq(path: Path, names: dict[str, str]) -> list[Chunk]:
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        product = names.get(item["product_id"], item["product_id"])
        body = f"Q: {item['question']}\nA: {item['answer']}"
        chunks.append(
            Chunk(
                chunk_id=f"faq#{item['id']}",
                text=f"{make_header(product, 'Frequently asked questions', item['question'])}\n\n{body}",
                metadata={
                    "doc_id": "faq",
                    "doc_type": "faq",
                    "title": "Frequently asked questions",
                    "product_id": item["product_id"],
                    "section": item["question"],
                    "tags": item.get("tags", []),
                    "source": "data/raw/faq/faq.jsonl",
                },
            )
        )
    return chunks


def load_chunks(raw_dir: Path = RAW_DIR) -> list[Chunk]:
    """Load every document listed in catalog.json and return all chunks."""
    catalog = load_catalog(raw_dir)
    names = product_names(catalog)
    chunks: list[Chunk] = []
    for doc in catalog["documents"]:
        path = raw_dir / doc["path"]
        if doc["format"] == "markdown":
            chunks.extend(chunk_markdown(path, names))
        elif doc["doc_type"] == "troubleshooting":
            chunks.extend(chunk_error_codes(path, names))
        elif doc["doc_type"] == "faq":
            chunks.extend(chunk_faq(path, names))
        else:
            raise ValueError(f"No loader for document {doc['doc_id']}")

    ids = [c.chunk_id for c in chunks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate chunk ids: {sorted(duplicates)}")
    return chunks
