from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Callable

from langchain.tools import ToolRuntime
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, tool

from agent.core.settings import Settings
from agent.logging.execution_log import ExecutionLog


JsonGetter = Callable[[str, dict[str, str]], Any]
TextGetter = Callable[[str, dict[str, str]], str]

_JSON_WORKSPACE_RE = re.compile(r'"assigned_external_workspace_abs"\s*:\s*"(?P<path>(?:\\.|[^"\\])*)"')
_KEYED_WORKSPACE_RE = re.compile(
    r"assigned_external_workspace_abs[`'\"]?\s*[:=]\s*[`'\"]?(?P<path>[^\n\r`'\"]+)[`'\"]?",
    re.IGNORECASE,
)
_ASSIGNED_WORKSPACE_RE = re.compile(
    r"assigned external workspace:\s*[`'\"]?(?P<path>[^\n\r`'\"]+)[`'\"]?",
    re.IGNORECASE,
)


class ExternalResearchClient:
    """Small HTTP clients for cutoff-bound external research APIs."""

    def __init__(
        self,
        settings: Settings,
        *,
        json_getter: JsonGetter | None = None,
        text_getter: TextGetter | None = None,
    ) -> None:
        self.settings = settings
        self._json_getter = json_getter or _http_get_json
        self._text_getter = text_getter or _http_get_text

    def search_semantic_scholar(
        self,
        *,
        query: str,
        cutoff: date,
        top_k: int,
        max_snippet_chars: int,
    ) -> dict[str, Any]:
        if not self.settings.semantic_scholar_proxy_api_key:
            return _result_payload(
                source="semantic_scholar",
                query=query,
                cutoff=cutoff,
                request=self._semantic_scholar_request(query=query, top_k=top_k, cutoff=cutoff),
                items=[],
                excluded_count=0,
                mode="missing_api_key",
                error="SEMANTIC_SCHOLAR_PROXY_API_KEY is required for Semantic Scholar proxy requests.",
            )

        request = self._semantic_scholar_request(query=query, top_k=top_k, cutoff=cutoff)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.settings.semantic_scholar_proxy_api_key}",
        }
        payload = self._json_getter(request["url"], headers)
        raw_items = payload.get("data") if isinstance(payload, dict) else []
        if not isinstance(raw_items, list):
            raw_items = []

        items: list[dict[str, Any]] = []
        excluded_count = 0
        for item in raw_items:
            if not isinstance(item, dict):
                excluded_count += 1
                continue
            publication_date = _parse_iso_date(item.get("publicationDate"))
            year = _safe_int(item.get("year"))
            if not _is_before_cutoff(publication_date=publication_date, year=year, cutoff=cutoff):
                excluded_count += 1
                continue
            items.append(
                {
                    "paper_id": item.get("paperId"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "abstract": _truncate_text(str(item.get("abstract") or ""), max_snippet_chars),
                    "tldr": _tldr_text(item.get("tldr"), max_snippet_chars),
                    "year": year,
                    "publication_date": publication_date.isoformat() if publication_date else None,
                    "citation_count": item.get("citationCount"),
                    "influential_citation_count": item.get("influentialCitationCount"),
                    "external_ids": item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {},
                    "open_access_pdf": item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else None,
                    "authors": _semantic_authors(item.get("authors")),
                }
            )
            if len(items) >= top_k:
                break

        return _result_payload(
            source="semantic_scholar",
            query=query,
            cutoff=cutoff,
            request=request,
            items=items,
            excluded_count=excluded_count + max(0, len(raw_items) - excluded_count - len(items)),
        )

    def search_arxiv(
        self,
        *,
        query: str,
        cutoff: date,
        max_results: int,
        max_snippet_chars: int,
        category: str | None = None,
    ) -> dict[str, Any]:
        request = self._arxiv_request(query=query, max_results=max_results, category=category)
        headers = {
            "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.8",
            "User-Agent": self.settings.arxiv_user_agent,
        }
        text = self._text_getter(request["url"], headers)
        raw_items = _parse_arxiv_entries(text)
        items: list[dict[str, Any]] = []
        excluded_count = 0
        for item in raw_items:
            published = _parse_iso_date(item.get("published"))
            if published is None or published >= cutoff:
                excluded_count += 1
                continue
            items.append(
                {
                    "arxiv_id": item.get("arxiv_id"),
                    "title": item.get("title"),
                    "summary": _truncate_text(str(item.get("summary") or ""), max_snippet_chars),
                    "authors": item.get("authors") or [],
                    "published": published.isoformat(),
                    "updated": item.get("updated"),
                    "primary_category": item.get("primary_category"),
                    "categories": item.get("categories") or [],
                    "links": item.get("links") or [],
                }
            )
            if len(items) >= max_results:
                break

        return _result_payload(
            source="arxiv",
            query=query,
            cutoff=cutoff,
            request=request,
            items=items,
            excluded_count=excluded_count + max(0, len(raw_items) - excluded_count - len(items)),
        )

    def search_github(
        self,
        *,
        query: str,
        cutoff: date,
        top_k: int,
        include_readme: bool,
        max_readme_chars: int,
    ) -> dict[str, Any]:
        request = self._github_repo_search_request(query=query, cutoff=cutoff, top_k=top_k)
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": self.settings.github_api_version,
            "User-Agent": self.settings.github_user_agent,
        }
        if self.settings.github_api_key:
            headers["Authorization"] = f"Bearer {self.settings.github_api_key}"

        payload = self._json_getter(request["url"], headers)
        raw_items = payload.get("items") if isinstance(payload, dict) else []
        if not isinstance(raw_items, list):
            raw_items = []

        items: list[dict[str, Any]] = []
        excluded_count = 0
        for repo in raw_items:
            if not isinstance(repo, dict):
                excluded_count += 1
                continue
            created_at = _parse_iso_date(repo.get("created_at"))
            full_name = repo.get("full_name")
            if not isinstance(full_name, str) or not full_name or created_at is None or created_at >= cutoff:
                excluded_count += 1
                continue

            commit = self._github_latest_commit_before_cutoff(full_name=full_name, cutoff=cutoff, headers=headers)
            if commit is None:
                excluded_count += 1
                continue

            item = {
                "full_name": full_name,
                "html_url": repo.get("html_url"),
                "description": repo.get("description"),
                "created_at": repo.get("created_at"),
                "pushed_at_current": repo.get("pushed_at"),
                "default_branch_current": repo.get("default_branch"),
                "language": repo.get("language"),
                "license": repo.get("license") if isinstance(repo.get("license"), dict) else None,
                "current_metadata": {
                    "stars": repo.get("stargazers_count"),
                    "forks": repo.get("forks_count"),
                    "open_issues": repo.get("open_issues_count"),
                },
                "snapshot_before_cutoff": commit,
            }
            if include_readme:
                item["readme_snapshot"] = self._github_readme_snapshot(
                    full_name=full_name,
                    sha=str(commit.get("sha") or ""),
                    headers=headers,
                    max_readme_chars=max_readme_chars,
                )
            items.append(item)
            if len(items) >= top_k:
                break

        return _result_payload(
            source="github",
            query=query,
            cutoff=cutoff,
            request=request,
            items=items,
            excluded_count=excluded_count + max(0, len(raw_items) - excluded_count - len(items)),
        )

    def _semantic_scholar_request(self, *, query: str, top_k: int, cutoff: date | None) -> dict[str, Any]:
        limit = _candidate_limit(top_k)
        params = {
            "query": query,
            "limit": str(limit),
            "fields": ",".join(
                [
                    "paperId",
                    "title",
                    "abstract",
                    "year",
                    "publicationDate",
                    "url",
                    "citationCount",
                    "influentialCitationCount",
                    "openAccessPdf",
                    "externalIds",
                    "authors",
                    "tldr",
                ]
            ),
        }
        if cutoff is not None:
            params["year"] = f"-{cutoff.year}"
        url = _url_with_params(_join_url(self.settings.semantic_scholar_proxy_base_url, "graph/v1/paper/search"), params)
        request = {"url": url, "method": "GET", "limit": limit, "fields": params["fields"]}
        if "year" in params:
            request["year"] = params["year"]
        return request

    def _arxiv_request(self, *, query: str, max_results: int, category: str | None) -> dict[str, Any]:
        candidate_query = query
        if category and category.strip():
            candidate_query = f"cat:{category.strip()} AND ({query})"
        limit = _candidate_limit(max_results)
        params = {
            "search_query": candidate_query,
            "start": "0",
            "max_results": str(limit),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        return {
            "url": _url_with_params(self.settings.arxiv_api_base_url, params),
            "method": "GET",
            "max_results": limit,
            "category": category,
        }

    def _github_repo_search_request(self, *, query: str, cutoff: date, top_k: int) -> dict[str, Any]:
        limit = _candidate_limit(top_k)
        search_query = f"{query} created:<{cutoff.isoformat()}"
        params = {"q": search_query, "per_page": str(limit), "sort": "stars", "order": "desc"}
        return {
            "url": _url_with_params(_join_url(self.settings.github_api_base_url, "search/repositories"), params),
            "method": "GET",
            "query": search_query,
            "per_page": limit,
        }

    def _github_latest_commit_before_cutoff(
        self,
        *,
        full_name: str,
        cutoff: date,
        headers: dict[str, str],
    ) -> dict[str, Any] | None:
        url = _url_with_params(
            _join_url(self.settings.github_api_base_url, f"repos/{full_name}/commits"),
            {"until": f"{cutoff.isoformat()}T00:00:00Z", "per_page": "1"},
        )
        try:
            payload = self._json_getter(url, headers)
        except Exception:
            return None
        if not isinstance(payload, list) or not payload:
            return None
        first = payload[0]
        if not isinstance(first, dict):
            return None
        commit = first.get("commit") if isinstance(first.get("commit"), dict) else {}
        committer = commit.get("committer") if isinstance(commit.get("committer"), dict) else {}
        authored = commit.get("author") if isinstance(commit.get("author"), dict) else {}
        commit_date = _parse_iso_date(committer.get("date") or authored.get("date"))
        if commit_date is None or commit_date >= cutoff:
            return None
        return {
            "sha": first.get("sha"),
            "html_url": first.get("html_url"),
            "commit_date": (committer.get("date") or authored.get("date")),
            "message": _truncate_text(str(commit.get("message") or ""), 300),
        }

    def _github_readme_snapshot(
        self,
        *,
        full_name: str,
        sha: str,
        headers: dict[str, str],
        max_readme_chars: int,
    ) -> dict[str, Any]:
        url = _url_with_params(
            _join_url(self.settings.github_api_base_url, f"repos/{full_name}/readme"),
            {"ref": sha},
        )
        try:
            payload = self._json_getter(url, headers)
        except Exception as exc:
            return {"available": False, "error": str(exc), "ref": sha}
        if not isinstance(payload, dict):
            return {"available": False, "error": "GitHub README response was not an object.", "ref": sha}
        content = payload.get("content")
        if not isinstance(content, str):
            return {"available": False, "error": "README content is missing.", "ref": sha}
        try:
            decoded = base64.b64decode(content.encode("ascii"), validate=False).decode("utf-8", errors="replace")
        except Exception as exc:
            return {"available": False, "error": f"README decode failed: {exc}", "ref": sha}
        return {
            "available": True,
            "name": payload.get("name"),
            "path": payload.get("path"),
            "html_url": payload.get("html_url"),
            "ref": sha,
            "text": _truncate_text(decoded, max_readme_chars),
        }


class ControlledExternalResearchTools:
    """Controlled external API tools for the reviewer-only external_researcher subagent."""

    def __init__(
        self,
        *,
        settings: Settings,
        run_root: Path | None = None,
        execution_log: ExecutionLog | None = None,
        client_factory: Callable[[Settings], ExternalResearchClient] = ExternalResearchClient,
    ) -> None:
        self.settings = settings
        self.run_root = run_root
        self.execution_log = execution_log
        self.client_factory = client_factory

    def to_tools(self) -> list[BaseTool]:
        search_semantic_scholar_impl = self.search_semantic_scholar
        search_arxiv_impl = self.search_arxiv
        search_github_impl = self.search_github

        @tool("search_semantic_scholar")
        def search_semantic_scholar_tool(
            query: Annotated[str, "Semantic Scholar search query. Do not pass the exact full current paper title."],
            output_filename: Annotated[str, "Relative .json artifact filename inside assigned_external_workspace_abs."],
            runtime: ToolRuntime,
            top_k: Annotated[int, "Maximum included results after cutoff filtering."] = 8,
            max_snippet_chars: Annotated[int, "Maximum abstract/TLDR characters saved per item."] = 800,
        ) -> str:
            """Search Semantic Scholar through the configured proxy with the env cutoff enforced."""
            return search_semantic_scholar_impl(
                state=runtime.state,
                config=runtime.config,
                query=query,
                output_filename=output_filename,
                top_k=top_k,
                max_snippet_chars=max_snippet_chars,
            )

        @tool("search_arxiv")
        def search_arxiv_tool(
            query: Annotated[str, "arXiv Atom API query. Prefer topic and method keywords, not the exact paper title."],
            output_filename: Annotated[str, "Relative .json artifact filename inside assigned_external_workspace_abs."],
            runtime: ToolRuntime,
            max_results: Annotated[int, "Maximum included results after cutoff filtering."] = 8,
            category: Annotated[str | None, "Optional arXiv category filter, for example cs.LG."] = None,
            max_snippet_chars: Annotated[int, "Maximum summary characters saved per item."] = 800,
        ) -> str:
            """Search arXiv Atom API with the env cutoff enforced against published dates."""
            return search_arxiv_impl(
                state=runtime.state,
                config=runtime.config,
                query=query,
                output_filename=output_filename,
                max_results=max_results,
                category=category,
                max_snippet_chars=max_snippet_chars,
            )

        @tool("search_github")
        def search_github_tool(
            query: Annotated[str, "GitHub repository search query. Prefer method/library keywords."],
            output_filename: Annotated[str, "Relative .json artifact filename inside assigned_external_workspace_abs."],
            runtime: ToolRuntime,
            top_k: Annotated[int, "Maximum included repositories after cutoff filtering."] = 8,
            include_readme: Annotated[bool, "Whether to include README text from the last pre-cutoff commit."] = False,
            max_readme_chars: Annotated[int, "Maximum README snapshot characters saved per repository."] = 1200,
        ) -> str:
            """Search GitHub repos and optionally read README snapshots at the last pre-cutoff commit."""
            return search_github_impl(
                state=runtime.state,
                config=runtime.config,
                query=query,
                output_filename=output_filename,
                top_k=top_k,
                include_readme=include_readme,
                max_readme_chars=max_readme_chars,
            )

        return [search_semantic_scholar_tool, search_arxiv_tool, search_github_tool]

    def search_semantic_scholar(
        self,
        *,
        state: Any,
        query: str,
        output_filename: str,
        top_k: int = 8,
        max_snippet_chars: int = 800,
        config: Any | None = None,
    ) -> str:
        return self._run_search(
            source="semantic_scholar",
            state=state,
            config=config,
            query=query,
            output_filename=output_filename,
            request_args={"top_k": top_k, "max_snippet_chars": max_snippet_chars},
            search=lambda client, cutoff: client.search_semantic_scholar(
                query=query,
                cutoff=cutoff,
                top_k=_bounded_count(top_k),
                max_snippet_chars=_bounded_chars(max_snippet_chars),
            ),
        )

    def search_arxiv(
        self,
        *,
        state: Any,
        query: str,
        output_filename: str,
        max_results: int = 8,
        category: str | None = None,
        max_snippet_chars: int = 800,
        config: Any | None = None,
    ) -> str:
        return self._run_search(
            source="arxiv",
            state=state,
            config=config,
            query=query,
            output_filename=output_filename,
            request_args={
                "max_results": max_results,
                "category": category,
                "max_snippet_chars": max_snippet_chars,
            },
            search=lambda client, cutoff: client.search_arxiv(
                query=query,
                cutoff=cutoff,
                max_results=_bounded_count(max_results),
                category=category,
                max_snippet_chars=_bounded_chars(max_snippet_chars),
            ),
        )

    def search_github(
        self,
        *,
        state: Any,
        query: str,
        output_filename: str,
        top_k: int = 8,
        include_readme: bool = False,
        max_readme_chars: int = 1200,
        config: Any | None = None,
    ) -> str:
        return self._run_search(
            source="github",
            state=state,
            config=config,
            query=query,
            output_filename=output_filename,
            request_args={
                "top_k": top_k,
                "include_readme": include_readme,
                "max_readme_chars": max_readme_chars,
            },
            search=lambda client, cutoff: client.search_github(
                query=query,
                cutoff=cutoff,
                top_k=_bounded_count(top_k),
                include_readme=bool(include_readme),
                max_readme_chars=_bounded_chars(max_readme_chars),
            ),
        )

    def _run_search(
        self,
        *,
        source: str,
        state: Any,
        config: Any | None,
        query: str,
        output_filename: str,
        request_args: dict[str, Any],
        search: Callable[[ExternalResearchClient, date], dict[str, Any]],
    ) -> str:
        workspace = _extract_external_workspace(state=state, config=config, run_root=self.run_root)
        ctx = self._timeline_context(config=config, workspace=workspace)
        cutoff, cutoff_error = _parse_cutoff(self.settings.external_research_cutoff_date)
        self._record_tool_call(
            ctx,
            f"search_{source}",
            {
                "query": _truncate_text(query, 240),
                "result_file": self._rel_workspace_value(workspace, output_filename),
                "cutoff_date": cutoff.isoformat() if cutoff else self.settings.external_research_cutoff_date,
                **_safe_request_args(request_args),
            },
        )

        output_path_or_error = _resolve_external_workspace_file(workspace, output_filename, required_suffix=".json")
        if isinstance(output_path_or_error, str):
            self._record_tool_result(ctx, f"search_{source}", {"error": output_path_or_error})
            return _json_error(output_path_or_error)

        if cutoff_error is not None or cutoff is None:
            result = _result_payload(
                source=source,
                query=query,
                cutoff=None,
                request={"source": source, **_safe_request_args(request_args)},
                items=[],
                excluded_count=0,
                mode=cutoff_error["mode"],
                error=cutoff_error["error"],
            )
        else:
            try:
                result = search(self.client_factory(self.settings), cutoff)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError, ET.ParseError) as exc:
                result = _result_payload(
                    source=source,
                    query=query,
                    cutoff=cutoff,
                    request={"source": source, **_safe_request_args(request_args)},
                    items=[],
                    excluded_count=0,
                    mode="request_error",
                    error=str(exc),
                )

        output_path_or_error.parent.mkdir(parents=True, exist_ok=True)
        output_path_or_error.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        items = result.get("items") if isinstance(result.get("items"), list) else []
        self._record_tool_result(
            ctx,
            f"search_{source}",
            {
                "result_file": self._rel(output_path_or_error),
                "item_count": len(items),
                "excluded_count": result.get("excluded_count", 0),
                "mode": result.get("mode"),
                "error": result.get("error"),
            },
        )
        return _json_success(
            action=f"search_{source}",
            output_path=str(output_path_or_error),
            source=source,
            mode=result.get("mode"),
            cutoff_date=result.get("cutoff_date"),
            item_count=len(items),
            excluded_count=result.get("excluded_count", 0),
            error=result.get("error"),
        )

    def _timeline_context(self, *, config: Any | None, workspace: Path | None) -> tuple[str | None, int | None]:
        configurable = _configurable(config)
        role = configurable.get("timeline_role")
        round_id = configurable.get("timeline_round")
        if not isinstance(role, str) or not role:
            role = self._role_from_workspace(workspace)
        if not isinstance(round_id, int):
            try:
                round_id = int(round_id) if round_id is not None else self._round_from_workspace(workspace)
            except (TypeError, ValueError):
                round_id = self._round_from_workspace(workspace)
        return role, round_id

    def _record_tool_call(self, ctx: tuple[str | None, int | None], name: str, input_payload: dict[str, Any]) -> None:
        if self.execution_log is None:
            return
        role, round_id = ctx
        self.execution_log.record(event="tool_call", role=role, round_id=round_id, name=name, input=input_payload)

    def _record_tool_result(self, ctx: tuple[str | None, int | None], name: str, output_payload: dict[str, Any]) -> None:
        if self.execution_log is None:
            return
        role, round_id = ctx
        self.execution_log.record(event="tool_result", role=role, round_id=round_id, name=name, output=output_payload)

    def _rel(self, path: Path) -> str:
        if self.execution_log is not None:
            return self.execution_log.rel(path)
        return str(path)

    def _rel_workspace_value(self, workspace: Path | None, value: str) -> str:
        if workspace is None:
            return value
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        return self._rel(candidate)

    @staticmethod
    def _role_from_workspace(workspace: Path | None) -> str | None:
        if workspace is None:
            return None
        name = workspace.name
        return name.rsplit("_round_", 1)[0] if "_round_" in name else None

    @staticmethod
    def _round_from_workspace(workspace: Path | None) -> int | None:
        if workspace is None:
            return None
        name = workspace.name
        if "_round_" not in name:
            return None
        try:
            return int(name.rsplit("_round_", 1)[1])
        except ValueError:
            return None


def build_external_research_tools(
    *,
    settings: Settings,
    run_root: Path | None = None,
    execution_log: ExecutionLog | None = None,
    client_factory: Callable[[Settings], ExternalResearchClient] = ExternalResearchClient,
) -> list[BaseTool]:
    return ControlledExternalResearchTools(
        settings=settings,
        run_root=run_root,
        execution_log=execution_log,
        client_factory=client_factory,
    ).to_tools()


def _http_get_json(url: str, headers: dict[str, str]) -> Any:
    text = _http_get_text(url, headers)
    return json.loads(text)


def _http_get_text(url: str, headers: dict[str, str]) -> str:
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _parse_cutoff(raw: str | None) -> tuple[date | None, dict[str, str] | None]:
    if raw is None or not str(raw).strip():
        return None, {
            "mode": "missing_cutoff_date",
            "error": "PRIORREVIEW_EXTERNAL_RESEARCH_CUTOFF_DATE is required before external research tools can run.",
        }
    try:
        return datetime.strptime(str(raw).strip(), "%Y-%m-%d").date(), None
    except ValueError:
        return None, {
            "mode": "invalid_cutoff_date",
            "error": "PRIORREVIEW_EXTERNAL_RESEARCH_CUTOFF_DATE must use YYYY-MM-DD format.",
        }


def _resolve_external_workspace_file(workspace: Path | None, value: str, *, required_suffix: str) -> Path | str:
    if workspace is None:
        return "assigned_external_workspace_abs is missing from the subagent task."
    if not value or not value.strip():
        return "Artifact filename is empty."
    candidate = Path(value.strip())
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved_workspace = workspace.expanduser().resolve(strict=False)
    resolved_candidate = candidate.expanduser().resolve(strict=False)
    if resolved_candidate != resolved_workspace and resolved_workspace not in resolved_candidate.parents:
        return f"Artifact path must stay inside assigned_external_workspace_abs: {value}"
    try:
        relative_parts = resolved_candidate.relative_to(resolved_workspace).parts
    except ValueError:
        return f"Artifact path must stay inside assigned_external_workspace_abs: {value}"
    if "subagent_traces" in relative_parts:
        return "Artifacts may not be written under subagent_traces."
    if resolved_candidate.suffix.lower() != required_suffix:
        return f"Artifact must use {required_suffix} suffix: {value}"
    return resolved_candidate


def _extract_external_workspace(*, state: Any, config: Any | None = None, run_root: Path | None = None) -> Path | None:
    workspace = _extract_external_workspace_from_config(config, run_root=run_root)
    if workspace is not None:
        return workspace.expanduser().resolve(strict=False)
    return _extract_external_workspace_from_state(state, run_root=run_root)


def _extract_external_workspace_from_config(config: Any | None, *, run_root: Path | None) -> Path | None:
    configurable = _configurable(config)
    abs_workspace = configurable.get("assigned_external_workspace_abs")
    if isinstance(abs_workspace, str) and abs_workspace.strip():
        candidate = Path(abs_workspace.strip()).expanduser()
        if candidate.is_absolute():
            return candidate.resolve(strict=False)

    workspace = configurable.get("assigned_external_workspace")
    if not isinstance(workspace, str) or not workspace.strip():
        return None

    candidate = Path(workspace.strip()).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
        if run_root is None or _is_relative_to(resolved, run_root.expanduser().resolve(strict=False)):
            return resolved
        return None

    if run_root is None:
        return None
    resolved_run_root = run_root.expanduser().resolve(strict=False)
    resolved = (resolved_run_root / candidate).resolve(strict=False)
    if _is_relative_to(resolved, resolved_run_root):
        return resolved
    return None


def _extract_external_workspace_from_state(state: Any, *, run_root: Path | None) -> Path | None:
    for message in _state_messages(state):
        workspace = _extract_external_workspace_from_text(_message_text(message), run_root=run_root)
        if workspace is not None:
            return workspace.expanduser().resolve(strict=False)
    return None


def _extract_external_workspace_from_text(text: str, *, run_root: Path | None) -> Path | None:
    json_match = _JSON_WORKSPACE_RE.search(text)
    if json_match:
        try:
            return Path(json.loads(f'"{json_match.group("path")}"'))
        except json.JSONDecodeError:
            return _workspace_path_from_string(json_match.group("path"), run_root=run_root)

    for pattern in (_KEYED_WORKSPACE_RE, _ASSIGNED_WORKSPACE_RE):
        match = pattern.search(text)
        if match:
            return _workspace_path_from_string(match.group("path"), run_root=run_root)
    return None


def _workspace_path_from_string(value: str, *, run_root: Path | None) -> Path | None:
    cleaned = value.strip().strip("`'\" ,")
    if not cleaned:
        return None
    candidate = Path(cleaned).expanduser()
    if candidate.is_absolute():
        return candidate
    if run_root is None:
        return None
    return run_root / candidate


def _configurable(config: Any | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    configurable = config.get("configurable")
    return configurable if isinstance(configurable, dict) else {}


def _is_relative_to(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _state_messages(state: Any) -> list[Any]:
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])
    return list(messages or [])


def _message_text(message: Any) -> str:
    if isinstance(message, BaseMessage):
        return _content_to_text(message.content)
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content") or message.get("data", {}).get("content")
    return _content_to_text(content)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_to_text(item) for item in content)
    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return str(content["text"])
        if isinstance(content.get("content"), str):
            return str(content["content"])
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _parse_arxiv_entries(text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(text)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    entries: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        links = []
        for link in entry.findall("atom:link", ns):
            links.append({key: link.attrib.get(key) for key in ("href", "rel", "type", "title") if link.attrib.get(key)})
        categories = [category.attrib.get("term") for category in entry.findall("atom:category", ns) if category.attrib.get("term")]
        authors = [
            _normalize_space(name.text or "")
            for name in entry.findall("atom:author/atom:name", ns)
            if name.text and name.text.strip()
        ]
        arxiv_id = _normalize_space(entry.findtext("atom:id", default="", namespaces=ns)).rsplit("/", 1)[-1]
        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": _normalize_space(entry.findtext("atom:title", default="", namespaces=ns)),
                "summary": _normalize_space(entry.findtext("atom:summary", default="", namespaces=ns)),
                "authors": authors,
                "published": entry.findtext("atom:published", default="", namespaces=ns),
                "updated": entry.findtext("atom:updated", default="", namespaces=ns),
                "primary_category": _primary_arxiv_category(entry, ns),
                "categories": categories,
                "links": links,
            }
        )
    return entries


def _primary_arxiv_category(entry: ET.Element, ns: dict[str, str]) -> str | None:
    primary = entry.find("arxiv:primary_category", ns)
    if primary is not None:
        return primary.attrib.get("term")
    category = entry.find("atom:category", ns)
    return category.attrib.get("term") if category is not None else None


def _result_payload(
    *,
    source: str,
    query: str,
    cutoff: date | None,
    request: dict[str, Any],
    items: list[dict[str, Any]],
    excluded_count: int,
    mode: str = "live",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "available": error is None,
        "source": source,
        "mode": mode,
        "query": query,
        "cutoff_date": cutoff.isoformat() if cutoff is not None else None,
        "request": _redact_request(request),
        "items": items,
        "excluded_count": excluded_count,
        "error": error,
    }


def _redact_request(request: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in request.items():
        if re.search(r"(api[_-]?key|token|secret|authorization|bearer)", str(key), re.IGNORECASE):
            redacted[str(key)] = "[REDACTED]"
        else:
            redacted[str(key)] = value
    return redacted


def _safe_request_args(args: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in args.items() if value is not None}


def _is_before_cutoff(*, publication_date: date | None, year: int | None, cutoff: date) -> bool:
    if publication_date is not None:
        return publication_date < cutoff
    if year is not None:
        return year < cutoff.year
    return False


def _parse_iso_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) < 4:
        return None
    if len(text) == 4 and text.isdigit():
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _semantic_authors(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    authors: list[dict[str, Any]] = []
    for author in value:
        if not isinstance(author, dict):
            continue
        authors.append({"author_id": author.get("authorId"), "name": author.get("name")})
    return authors


def _tldr_text(value: Any, max_chars: int) -> str | None:
    if not isinstance(value, dict):
        return None
    text = value.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return _truncate_text(text, max_chars)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bounded_count(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 8
    return min(25, max(1, parsed))


def _bounded_chars(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 800
    return min(4000, max(120, parsed))


def _candidate_limit(value: int) -> int:
    return min(100, max(_bounded_count(value), _bounded_count(value) * 4))


def _url_with_params(url: str, params: dict[str, str]) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urllib.parse.urlencode(params)}"


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _json_success(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def _json_error(error: str) -> str:
    return json.dumps({"ok": False, "error": error}, ensure_ascii=False)


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _truncate_text(text: str, max_chars: int) -> str:
    clean = _normalize_space(text)
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."
