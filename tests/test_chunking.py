"""Unit tests for dataset chunking (run on the real data/raw files, no API calls)."""

from __future__ import annotations

import unittest

from orbitra_rag.chunking import load_chunks, parse_front_matter, split_long, split_sections


class ParsingTest(unittest.TestCase):
    def test_front_matter(self):
        meta, body = parse_front_matter('---\ndoc_id: x\nversion: "1.2"\n---\n# Title\ntext')
        self.assertEqual(meta, {"doc_id": "x", "version": "1.2"})
        self.assertIn("# Title", body)

    def test_split_sections_with_subsections(self):
        body = "# Title\nintro\n## A\na text\n## B\n### B1\nb1 text\n## Empty\n"
        self.assertEqual(split_sections(body), [("A", "a text"), ("B › B1", "b1 text")])

    def test_split_long_respects_paragraphs(self):
        parts = split_long("aaaa\n\nbbbb\n\ncccc", limit=10)
        self.assertEqual(parts, ["aaaa\n\nbbbb", "cccc"])


class DatasetChunksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_chunks()
        cls.by_id = {c.chunk_id: c for c in cls.chunks}

    def test_counts_per_type(self):
        counts = {}
        for c in self.chunks:
            counts[c.metadata["doc_type"]] = counts.get(c.metadata["doc_type"], 0) + 1
        self.assertEqual(counts["troubleshooting"], 27)
        self.assertEqual(counts["faq"], 23)
        self.assertEqual(counts["release_notes"], 7)
        self.assertGreater(counts["manual"], 40)

    def test_required_metadata(self):
        for c in self.chunks:
            for key in ("doc_id", "doc_type", "product_id", "section", "title", "source"):
                self.assertTrue(c.metadata.get(key), f"{c.chunk_id} has no {key}")

    def test_chunk_text_has_context_header(self):
        for c in self.chunks:
            self.assertTrue(c.text.startswith("Product: "), c.chunk_id)

    def test_error_code_chunk(self):
        chunk = self.by_id["error-codes#robovac-r7-e06"]
        self.assertEqual(chunk.metadata["code"], "E06")
        self.assertIn("LiDAR", chunk.text)
        self.assertIn("Orbitra RoboVac R7", chunk.text)

    def test_release_notes_get_their_own_product(self):
        chunk = self.by_id["release-notes-firmware#thermosmart-t2-3-1-0-2026-09-01"]
        self.assertEqual(chunk.metadata["product_id"], "thermosmart-t2")
        self.assertEqual(chunk.metadata["firmware_version"], "3.1.0")

    def test_policies_are_cross_product(self):
        policy = [c for c in self.chunks if c.metadata["doc_type"] == "policy"]
        self.assertTrue(policy)
        self.assertTrue(all(c.metadata["product_id"] == "all" for c in policy))


if __name__ == "__main__":
    unittest.main()
