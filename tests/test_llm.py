"""Offline unit tests for LLMClient (the chat API is replaced by a fake)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from orbitra_rag.llm import LLMClient


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))]) for p in ("Hel", "lo", None)]
            )
        message = SimpleNamespace(content="  Hello!  ")
        return SimpleNamespace(
            model="fake-model",
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )


def make_llm(**kwargs):
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLMClient(client=client, **kwargs), completions


class LLMClientTest(unittest.TestCase):
    def test_ask_returns_text_and_usage(self):
        llm, api = make_llm()
        response = llm.ask("Hi", system="Be brief")
        self.assertEqual(response.text, "Hello!")
        self.assertEqual(response.usage["total_tokens"], 7)
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(
            api.calls[0]["messages"],
            [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hi"}],
        )

    def test_request_parameters(self):
        llm, api = make_llm(temperature=0.3, max_tokens=100)
        llm.ask("Hi")
        self.assertEqual(api.calls[0]["temperature"], 0.3)
        self.assertEqual(api.calls[0]["max_tokens"], 100)

    def test_max_tokens_omitted_by_default(self):
        llm, api = make_llm()
        llm.ask("Hi")
        self.assertNotIn("max_tokens", api.calls[0])

    def test_stream_skips_empty_deltas(self):
        llm, _ = make_llm()
        self.assertEqual("".join(llm.stream([{"role": "user", "content": "Hi"}])), "Hello")

    def test_empty_prompt_rejected(self):
        llm, _ = make_llm()
        with self.assertRaises(ValueError):
            llm.ask("   ")


if __name__ == "__main__":
    unittest.main()
