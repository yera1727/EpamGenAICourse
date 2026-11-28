"""Step 6: check the LLM client with simple request/response interactions.

Usage:
    python scripts/check_llm.py                 # two demo prompts
    python scripts/check_llm.py "your question" # one custom prompt
    python scripts/check_llm.py --chat          # interactive chat in the terminal (empty line to exit)

The second demo prompt asks about an Orbitra product WITHOUT any retrieved context: the model does
not know these fictional products, which shows why the RAG part is needed.
"""

from __future__ import annotations

import sys

from orbitra_rag.llm import LLMClient
from orbitra_rag.providers import ConfigurationError

DEMO_PROMPTS = [
    "In one sentence: what is Retrieval-Augmented Generation?",
    "What does error E06 mean on the Orbitra RoboVac R7 robot vacuum, and how do I fix it?",
]


def show(llm: LLMClient, prompt: str) -> None:
    response = llm.ask(prompt)
    print(f"\n> {prompt}\n{response.text}")
    print(f"  [{response.model}, {response.latency_s:.2f}s, tokens: {response.usage}, finish: {response.finish_reason}]")


def chat_loop(llm: LLMClient) -> None:
    print("Interactive chat. Empty line to exit.")
    history = [{"role": "system", "content": "You are a concise, helpful assistant."}]
    while True:
        try:
            user = input("\nyou> ").strip()
        except EOFError:
            break
        if not user:
            break
        history.append({"role": "user", "content": user})
        print("bot> ", end="", flush=True)
        answer = "".join(print(piece, end="", flush=True) or piece for piece in llm.stream(history))
        print()
        history.append({"role": "assistant", "content": answer})


def main() -> int:
    try:
        llm = LLMClient()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 1
    print(f"Provider: {llm.settings.provider} | chat model: {llm.model}")

    if "--chat" in sys.argv:
        chat_loop(llm)
    elif len(sys.argv) > 1:
        show(llm, " ".join(sys.argv[1:]))
    else:
        for prompt in DEMO_PROMPTS:
            show(llm, prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
