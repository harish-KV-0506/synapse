"""Optional LLM access. If no key / SDK, `llm.available` is False and agents use heuristics."""
from __future__ import annotations

import json
import logging
import re

from .config import settings

log = logging.getLogger("synapse.llm")


class LLM:
    def __init__(self):
        self.client = None
        if settings.anthropic_api_key:
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            except Exception as e:
                log.warning("anthropic SDK unavailable: %s", e)

    @property
    def available(self) -> bool:
        return self.client is not None

    def complete(self, system: str, prompt: str, max_tokens: int = 2500) -> str:
        assert self.client is not None
        msg = self.client.messages.create(
            model=settings.llm_model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}], temperature=0,
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    def json(self, system: str, prompt: str, max_tokens: int = 3000):
        raw = self.complete(system + "\nRespond with a single valid JSON object only.", prompt, max_tokens)
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else None


llm = LLM()
