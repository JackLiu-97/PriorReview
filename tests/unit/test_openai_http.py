from __future__ import annotations

import unittest

import httpx
from openai import InternalServerError

from agent.core.openai_http import (
    make_openai_http_client,
    openai_retry_directive,
    should_bypass_env_proxy,
)


def _internal_server_error(
    *,
    status_code: int = 502,
    body: dict | None = None,
    headers: dict | None = None,
) -> InternalServerError:
    response = httpx.Response(
        status_code,
        json=body or {},
        headers=headers or {},
        request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
    )
    return InternalServerError(
        f"Error code: {status_code}",
        response=response,
        body=body,
    )


class OpenAIHTTPTest(unittest.TestCase):
    def test_bypasses_env_proxy_for_loopback_urls(self) -> None:
        self.assertTrue(should_bypass_env_proxy("http://127.0.0.1:20128/v1"))
        self.assertTrue(should_bypass_env_proxy("http://localhost:20128/v1"))
        self.assertTrue(should_bypass_env_proxy("http://[::1]:20128/v1"))

    def test_keeps_default_proxy_behavior_for_remote_urls(self) -> None:
        self.assertFalse(should_bypass_env_proxy("https://api.openai.com/v1"))
        self.assertFalse(should_bypass_env_proxy("https://openrouter.ai/api/v1"))
        self.assertFalse(should_bypass_env_proxy(None))

    def test_loopback_http_client_disables_trust_env(self) -> None:
        client = make_openai_http_client("http://127.0.0.1:20128/v1")
        try:
            self.assertIsNotNone(client)
            self.assertFalse(client.trust_env)
        finally:
            if client is not None:
                client.close()

    def test_retry_directive_uses_body_retry_after(self) -> None:
        exc = _internal_server_error(body={"retryable": True, "retry_after": 60})

        directive = openai_retry_directive(exc, attempt=1)

        self.assertTrue(directive.retryable_http_error)
        self.assertEqual(directive.sleep_seconds, 60)

    def test_retry_directive_uses_larger_explicit_retry_after(self) -> None:
        exc = _internal_server_error(
            body={"retryable": True, "retry_after": 60},
            headers={"Retry-After": "30"},
        )

        directive = openai_retry_directive(exc, attempt=1)

        self.assertTrue(directive.retryable_http_error)
        self.assertEqual(directive.sleep_seconds, 60)

    def test_retry_directive_defaults_to_exponential_backoff(self) -> None:
        exc = _internal_server_error(body={})

        first = openai_retry_directive(exc, attempt=1)
        second = openai_retry_directive(exc, attempt=2)

        self.assertTrue(first.retryable_http_error)
        self.assertEqual(first.sleep_seconds, 5)
        self.assertEqual(second.sleep_seconds, 10)

    def test_retry_directive_ignores_non_http_errors(self) -> None:
        directive = openai_retry_directive(RuntimeError("format failure"), attempt=1)

        self.assertFalse(directive.retryable_http_error)
        self.assertEqual(directive.sleep_seconds, 0)


if __name__ == "__main__":
    unittest.main()
