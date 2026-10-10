from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlparse

import httpx

RETRYABLE_OPENAI_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0
MAX_RETRY_BACKOFF_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class RetryDirective:
    retryable_http_error: bool
    sleep_seconds: float


def should_bypass_env_proxy(base_url: str | None) -> bool:
    """Return True for loopback OpenAI-compatible endpoints."""

    if not base_url:
        return False
    host = urlparse(base_url).hostname
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def make_openai_http_client(base_url: str | None) -> httpx.Client | None:
    """Create a direct client for local proxies so system proxy settings do not intercept them."""

    if not should_bypass_env_proxy(base_url):
        return None
    return httpx.Client(trust_env=False)


def openai_retry_directive(exc: Exception, *, attempt: int) -> RetryDirective:
    """Return retry delay metadata for retryable OpenAI-compatible HTTP errors."""

    status_code = _status_code(exc)
    retryable_hint = _body_retryable_hint(exc)
    is_retryable_http_error = (
        status_code in RETRYABLE_OPENAI_STATUS_CODES
        and retryable_hint is not False
    )
    if not is_retryable_http_error:
        return RetryDirective(retryable_http_error=False, sleep_seconds=0.0)

    explicit_delays = [
        delay
        for delay in (
            _body_retry_after(exc),
            _header_retry_after(exc),
        )
        if delay is not None
    ]
    if explicit_delays:
        return RetryDirective(
            retryable_http_error=True,
            sleep_seconds=max(explicit_delays),
        )

    backoff = DEFAULT_RETRY_BACKOFF_SECONDS * (2 ** max(0, attempt - 1))
    return RetryDirective(
        retryable_http_error=True,
        sleep_seconds=min(backoff, MAX_RETRY_BACKOFF_SECONDS),
    )


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status_code = getattr(response, "status_code", None)
    return response_status_code if isinstance(response_status_code, int) else None


def _body_retryable_hint(exc: Exception) -> bool | None:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    retryable = body.get("retryable")
    return retryable if isinstance(retryable, bool) else None


def _body_retry_after(exc: Exception) -> float | None:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    return _parse_retry_after_seconds(body.get("retry_after"))


def _header_retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    return _parse_retry_after_seconds(headers.get("Retry-After"))


def _parse_retry_after_seconds(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return seconds
