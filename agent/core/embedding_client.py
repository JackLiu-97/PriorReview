#!/usr/bin/env python3
"""
OpenAI embedding 客户端。
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable

from dotenv import load_dotenv
from openai import OpenAI

from agent.core.openai_http import make_openai_http_client


load_dotenv()

DEFAULT_API_KEY = os.getenv("OPENAI_EMBEDDING_API_KEY")
DEFAULT_BASE_URL = os.getenv("OPENAI_EMBEDDING_BASE_URL")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL") or "openai/text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
DEFAULT_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")
DEFAULT_OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE") or os.getenv("OPENROUTER_APP_TITLE")
MODEL_DEFAULT_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "qwen/qwen3-embedding-8b": 4096,
    "Qwen/Qwen3-Embedding-8B": 4096,
}


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = " ".join(value.replace("\x00", "").split())
    return text or None


def vector_to_pg(vector: list[float] | None) -> str | None:
    if vector is None:
        return None
    return json.dumps(vector, ensure_ascii=False, separators=(",", ":"))


class OpenAIEmbeddingClient:
    """面向短文本 embedding 的轻量包装。"""

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int | None = None,
        batch_size: int = 256,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
    ):
        resolved_api_key = api_key or DEFAULT_API_KEY
        if not resolved_api_key:
            raise RuntimeError("Embedding API key is required. Set OPENAI_EMBEDDING_API_KEY in .env.")
        resolved_base_url = base_url or DEFAULT_BASE_URL
        if not resolved_base_url:
            raise RuntimeError("Embedding base URL is required. Set OPENAI_EMBEDDING_BASE_URL in .env.")

        self.model = model
        if dimensions is None and DEFAULT_EMBEDDING_DIMENSIONS:
            dimensions = int(DEFAULT_EMBEDDING_DIMENSIONS)
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max(1, max_retries)
        self.retry_base_delay = max(0.0, retry_base_delay)
        default_headers: dict[str, str] = {}
        if DEFAULT_HTTP_REFERER:
            default_headers["HTTP-Referer"] = DEFAULT_HTTP_REFERER
        if DEFAULT_OPENROUTER_TITLE:
            default_headers["X-Title"] = DEFAULT_OPENROUTER_TITLE

        client_kwargs: dict[str, object] = {
            "api_key": resolved_api_key,
            "base_url": resolved_base_url,
            "max_retries": 0,
        }
        http_client = make_openai_http_client(resolved_base_url)
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self.client = OpenAI(**client_kwargs)

    @property
    def resolved_dimensions(self) -> int | None:
        return self.dimensions or MODEL_DEFAULT_DIMS.get(self.model)

    def _create_embeddings_with_retry(
        self,
        *,
        kwargs: dict[str, object],
        batch: list[str],
        stage_name: str,
        batch_index: int,
        total_batches: int,
        logger: logging.Logger | None,
    ):
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.embeddings.create(**kwargs)
                if not getattr(response, "data", None):
                    raise ValueError("No embedding data received")
                if len(response.data) != len(batch):
                    raise ValueError(
                        f"Embedding count mismatch: expected {len(batch)}, got {len(response.data)}"
                    )
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries:
                    if logger is not None:
                        logger.error(
                            "Embedding request failed after %s attempts stage=%s batch=%s/%s size=%s error=%s",
                            self.max_retries,
                            stage_name,
                            batch_index,
                            total_batches,
                            len(batch),
                            exc,
                        )
                    raise

                delay_seconds = min(self.retry_base_delay * (2 ** (attempt - 1)), 8.0)
                if logger is not None:
                    logger.warning(
                        "Embedding request retry %s/%s stage=%s batch=%s/%s size=%s after error=%s; sleeping %.1fs",
                        attempt,
                        self.max_retries,
                        stage_name,
                        batch_index,
                        total_batches,
                        len(batch),
                        exc,
                        delay_seconds,
                    )
                time.sleep(delay_seconds)

        raise RuntimeError("Embedding request failed unexpectedly") from last_error

    def embed_missing(
        self,
        texts: Iterable[str | None],
        store: dict[str, list[float]],
        *,
        stage_name: str = "default",
        logger: logging.Logger | None = None,
    ) -> dict[str, int]:
        """只为当前批次中缺失的文本创建 embedding。"""
        normalized = [normalize_text(text) for text in texts]
        unique_texts = [text for text in dict.fromkeys(normalized) if text]
        missing = [text for text in unique_texts if text not in store]
        stats = {
            "requested": len(unique_texts),
            "reused": len(unique_texts) - len(missing),
            "missing": len(missing),
            "api_batches": 0,
        }
        if logger is not None:
            logger.info(
                "Embedding stage=%s requested=%s reused_in_batch=%s missing=%s",
                stage_name,
                stats["requested"],
                stats["reused"],
                stats["missing"],
            )
        if not missing:
            return stats

        total_batches = (len(missing) + self.batch_size - 1) // self.batch_size
        for batch_index, start in enumerate(range(0, len(missing), self.batch_size), start=1):
            batch = missing[start : start + self.batch_size]
            kwargs = {
                "model": self.model,
                "input": batch,
            }
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions

            if logger is not None:
                logger.info(
                    "Embedding request stage=%s batch=%s/%s size=%s model=%s dimensions=%s",
                    stage_name,
                    batch_index,
                    total_batches,
                    len(batch),
                    self.model,
                    self.resolved_dimensions,
                )
            response = self._create_embeddings_with_retry(
                kwargs=kwargs,
                batch=batch,
                stage_name=stage_name,
                batch_index=batch_index,
                total_batches=total_batches,
                logger=logger,
            )
            for item, text in zip(response.data, batch, strict=True):
                store[text] = item.embedding
            stats["api_batches"] += 1
            if logger is not None:
                logger.info(
                    "Embedding response stage=%s batch=%s/%s size=%s batch_unique_vectors=%s",
                    stage_name,
                    batch_index,
                    total_batches,
                    len(batch),
                    len(store),
                )
        return stats
