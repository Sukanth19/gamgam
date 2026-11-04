"""Shared HTTP helper: polite throttling + retries with exponential backoff.

Used by every API client (RAWG, IGDB, ...) so retry behaviour lives in one place.
"""

import logging
import random
import time

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504, 522, 524}


class ThrottledHttp:
    def __init__(self, *, base_url: str, min_interval: float, max_retries: int,
                 headers: dict | None = None, transport: httpx.BaseTransport | None = None):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_request = 0.0
        self.http = httpx.Client(base_url=base_url, timeout=30, transport=transport,
                                 headers={"User-Agent": "gamgam-learning-project", **(headers or {})})

    def close(self):
        self.http.close()

    def _throttle(self):
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        err = "unknown error"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.http.request(method, path, **kwargs)
            except httpx.TransportError as e:   # timeouts, connection resets...
                err = f"network error: {e}"
                retry_after = None
            else:
                if resp.status_code < 400:
                    return resp
                if resp.status_code not in RETRYABLE_STATUS:
                    resp.raise_for_status()      # 400/401/404: retrying won't help
                err = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")
            if attempt == self.max_retries:
                break
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)       # server told us how long to wait
            else:
                delay = min(60, 2 ** attempt) + random.uniform(0, 1)  # exp backoff + jitter
            log.warning("%s on %s, retry %d/%d in %.1fs", err, path,
                        attempt + 1, self.max_retries, delay)
            time.sleep(delay)
        raise RuntimeError(f"{method} {path} failed after {self.max_retries} retries: {err}")
