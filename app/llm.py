"""Shared LLM helper: ask for JSON matching a schema, never raise.

Two providers (LLM_PROVIDER in .env):
- ollama (default): a local model served by Ollama on this machine.
- gemini: Google's API; free-tier models are often busy (503) or rate-limited
  (429), so we try each model in config.CHAT_MODELS in order.

Callers just do:  client = make_client();  out = generate_json(client, prompt, Schema)
and get a parsed pydantic object back, or None if nothing worked (they then
degrade gracefully, e.g. show sources without an answer).
"""

import logging
import time

import httpx
from pydantic import TypeAdapter, ValidationError

from app import config

log = logging.getLogger(__name__)


class OllamaClient:
    """Minimal client for Ollama's /api/chat endpoint (https://docs.ollama.com/api).

    `format` = a JSON schema makes Ollama constrain the model's output to that
    schema. We still validate with pydantic: a local model can produce JSON
    that's valid but semantically off (we retry), and validation is cheap.
    """

    def __init__(self, url: str = config.OLLAMA_URL, model: str = config.OLLAMA_MODEL,
                 transport: httpx.BaseTransport | None = None):
        self.model = model
        self.http = httpx.Client(base_url=url, timeout=config.OLLAMA_TIMEOUT_S, transport=transport)

    def available(self) -> bool:
        try:
            tags = self.http.get("/api/tags", timeout=5).json()
        except httpx.HTTPError:
            log.warning("Ollama isn't reachable at %s (is the service running?)", self.http.base_url)
            return False
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            log.warning("Ollama model %r not pulled (have: %s). Run: ollama pull %s",
                        self.model, ", ".join(sorted(names)) or "none", self.model)
            return False
        return True

    def generate_json(self, prompt: str, schema, temperature: float):
        adapter = TypeAdapter(schema)
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": adapter.json_schema(),
            "stream": False,
            "think": False,      # qwen3 "thinking" mode: slow, not needed for short JSON tasks
            "options": {"temperature": temperature, "num_ctx": config.OLLAMA_NUM_CTX},
        }
        for attempt in range(config.OLLAMA_RETRIES):
            resp = self.http.post("/api/chat", json=body)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            try:
                return adapter.validate_json(content)
            except ValidationError as e:
                log.warning("ollama output didn't match the schema (attempt %d/%d): %s",
                            attempt + 1, config.OLLAMA_RETRIES, str(e).splitlines()[0])
        return None


def make_client(provider: str | None = None):
    """The configured LLM client, or None if it isn't usable (callers degrade).
    provider: "ollama" | "gemini"; defaults to LLM_PROVIDER."""
    provider = provider or config.LLM_PROVIDER
    if provider == "ollama":
        client = OllamaClient()
        return client if client.available() else None
    if not config.GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY not set")
        return None
    from google import genai
    from google.genai import types
    # Without a timeout one stalled free-tier model can hang a request for minutes;
    # with it, we give up on that model and fall through to the next one.
    return genai.Client(api_key=config.GEMINI_API_KEY,
                        http_options=types.HttpOptions(timeout=config.CHAT_TIMEOUT_S * 1000))


def generate_json(client, prompt: str, schema, *, temperature: float = 0.4, what: str = "response"):
    """Ask for JSON matching `schema` (a pydantic model or list[...] of one).
    Returns the parsed object, or None if it failed."""
    if isinstance(client, OllamaClient):
        try:
            out = client.generate_json(prompt, schema, temperature)
            if out is not None:
                log.info("%s from ollama/%s", what, client.model)
            return out
        except (httpx.HTTPError, KeyError, ValueError) as e:
            log.warning("ollama failed: %s", e)
            return None
    return _gemini_generate_json(client, prompt, schema, temperature, what)


def _gemini_generate_json(client, prompt, schema, temperature, what):
    from google.genai import errors, types
    cfg = types.GenerateContentConfig(response_mime_type="application/json",
                                      response_schema=schema, temperature=temperature)
    for model in config.CHAT_MODELS:
        for attempt in range(config.CHAT_RETRIES_PER_MODEL):
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                log.info("%s from %s", what, model)
                return resp.parsed
            except errors.APIError as e:
                if e.code not in (429, 500, 503):
                    log.warning("%s: %s", model, e)
                    break                           # this model won't work; try the next
                log.warning("%s busy (%s), attempt %d/%d", model, e.code,
                            attempt + 1, config.CHAT_RETRIES_PER_MODEL)
                time.sleep(2 * (attempt + 1))
            except Exception as e:                  # network errors etc.
                log.warning("%s failed: %s", model, e)
                break
    return None
