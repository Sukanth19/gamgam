"""Shared Gemini chat helper: structured (JSON) output with model fallback.

Free-tier models are often busy (503) or rate-limited (429), so we try each
model in config.CHAT_MODELS in order, with a couple of short retries each.
Returns None instead of raising when nothing works: callers degrade gracefully.
"""

import logging
import time

from app import config

log = logging.getLogger(__name__)


def make_client():
    """A Gemini client, or None if no API key is configured."""
    if not config.GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY not set")
        return None
    from google import genai
    return genai.Client(api_key=config.GEMINI_API_KEY)


def generate_json(client, prompt: str, schema, *, temperature: float = 0.4, what: str = "response"):
    """Ask for JSON matching `schema` (a pydantic model or list[...] of one).
    Returns the parsed object, or None if every model failed."""
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
