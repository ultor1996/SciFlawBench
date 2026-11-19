"""Building snapshot rows from raw live responses, and writing a snapshot directory."""

from __future__ import annotations

import html as html_lib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from . import schema
from .canon import canonical_arxiv_id, canonical_url, canonical_wiki_title, wiki_key

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_MD_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_MD_MARKUP = re.compile(r"[#*_`>|\[\]]+")


def _now() -> datetime:
    return datetime.now(UTC)


def _ts(value) -> datetime:
    if value is None:
        return _now()
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def plain_text(markdown: str, limit: int = 200_000) -> str:
    """Searchable text from a markdown rendering: link targets and markup removed."""
    text = _MD_LINK_TARGET.sub("]", markdown or "")
    text = _MD_MARKUP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def html_plain_text(body: str, limit: int = 200_000) -> str:
    """Searchable text straight from HTML (iterative; works on any nesting depth)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(body or "", "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()[:limit]


def html_title(body: str) -> str:
    match = _TITLE_RE.search(body or "")
    return re.sub(r"\s+", " ", html_lib.unescape(match.group(1))).strip() if match else ""


def _row(service: str, key: str, **fields) -> dict[str, Any]:
    row = dict.fromkeys(schema.SNAPSHOT_COLUMNS)
    row.update(
        doc_id=schema.doc_id(service, key),
        service=service,
        request_key=key,
        status=200,
        content_type="application/json; charset=utf-8",
        body="",
        markdown="",
        title="",
        snippet="",
        search_text="",
    )
    row.update(fields)
    row["fetched_at"] = _ts(fields.get("fetched_at"))
    return row


def web_row(
    url: str,
    status: int,
    content_type: str,
    body: str,
    *,
    fetched_at=None,
    snippet: str = "",
    title: str | None = None,
    lang: str | None = None,
) -> dict:
    """A page as `visit_webpage` sees it (markdown computed with the pinned markdownify)."""
    from .services import RENDER_ERROR_PREFIX, MarkdownRenderError, render_markdown

    markdown, search_text = "", ""
    if int(status) == 200:
        try:
            markdown = render_markdown(body)
            search_text = plain_text(markdown)
        except MarkdownRenderError as exc:
            # The live `visit_webpage` cannot read this page either; keep the page
            # searchable (search engines index it) but record the tool's failure.
            markdown = RENDER_ERROR_PREFIX + str(exc)
            search_text = html_plain_text(body)
    page_title = title if title is not None else html_title(body)
    return _row(
        schema.WEB,
        canonical_url(url),
        url=url,
        lang=lang,
        status=int(status),
        content_type=content_type,
        body=body,
        markdown=markdown,
        title=page_title or url,
        snippet=snippet or "",
        search_text=search_text,
        fetched_at=fetched_at,
    )


def wikipedia_row(
    lang: str, info: dict, extracts: dict, redirects: list[str] | None = None, *, fetched_at=None
) -> dict:
    """A Wikipedia article: raw `prop=info` and `prop=extracts` page objects + redirects."""
    title = canonical_wiki_title(info.get("title") or extracts.get("title") or "")
    body = {"info": info, "extracts": extracts, "redirects": sorted(set(redirects or []))}
    text = extracts.get("extract") or ""
    return _row(
        schema.WIKIPEDIA,
        wiki_key(lang, title),
        url=info.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
        lang=lang,
        body=json.dumps(body, ensure_ascii=False),
        title=title,
        search_text=re.sub(r"\s+", " ", text).strip(),
        fetched_at=fetched_at,
    )


def arxiv_row(arxiv_id: str, feed_xml: str, *, fetched_at=None) -> dict:
    """An arXiv paper: the raw Atom feed returned for `id_list=<id>`."""
    from .services import _arxiv_fields, _first_atom_entry

    entry = _first_atom_entry(feed_xml)
    fields = _arxiv_fields(entry) if entry is not None else {"ti": "", "abs": "", "au": ""}
    key = canonical_arxiv_id(arxiv_id)
    return _row(
        schema.ARXIV,
        key,
        url=f"https://arxiv.org/abs/{key}",
        lang="en",
        content_type="application/atom+xml; charset=utf-8",
        body=feed_xml,
        title=fields["ti"],
        search_text=" ".join([fields["ti"], fields["au"], fields["abs"]]).strip(),
        fetched_at=fetched_at,
    )


def golden_row(kind: str, service: str, url: str, status: int, content_type: str, body: str, fetched_at=None) -> dict:
    return {
        "kind": kind,
        "service": service,
        "url": url,
        "status": int(status),
        "content_type": content_type,
        "body": body,
        "fetched_at": _ts(fetched_at),
    }


def provenance_row(
    row: dict,
    role: str,
    task_ids=(),
    source_url: str = "",
    note: str = "",
    leak_flag: bool = False,
    leak_context: str = "",
) -> dict:
    return {
        "doc_id": row["doc_id"],
        "service": row["service"],
        "request_key": row["request_key"],
        "role": role,
        "task_ids": sorted(set(task_ids)),
        "source_url": source_url,
        "note": note,
        "leak_flag": bool(leak_flag),
        "leak_context": leak_context,
    }


def library_versions() -> dict[str, str]:
    out = {}
    for name in (
        "markdownify",
        "wikipedia-api",
        "arxiv",
        "requests",
        "pyarrow",
        "fastapi",
    ):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def _write_table(path: Path, rows: list[dict], columns: dict[str, str], sort_keys) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = sorted(rows, key=sort_keys)
    table = pa.Table.from_pylist(rows, schema=schema.arrow_schema(columns))
    pq.write_table(table, path, compression="zstd", use_dictionary=True, write_statistics=False)


def write_snapshot(
    out_dir: str | Path,
    rows: list[dict],
    provenance: list[dict] | None = None,
    golden: list[dict] | None = None,
    *,
    snapshot_date: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write snapshot.parquet (+ provenance/golden) and manifest.json; return the manifest.

    Rows are sorted by `doc_id` (provenance by `doc_id`, role; golden by kind, url) so the
    same inputs always produce byte-identical files.
    """
    schema.require_local_extra("pyarrow")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seen: dict[tuple[str, str], dict] = {}
    for row in rows:
        seen[(row["service"], row["request_key"])] = row  # last write wins
    rows = list(seen.values())

    _write_table(out / schema.SNAPSHOT_FILE, rows, schema.SNAPSHOT_COLUMNS, lambda r: r["doc_id"])
    files = [schema.SNAPSHOT_FILE]
    if provenance is not None:
        _write_table(
            out / schema.PROVENANCE_FILE,
            provenance,
            schema.PROVENANCE_COLUMNS,
            lambda r: (r["doc_id"], r["role"], r["source_url"] or ""),
        )
        files.append(schema.PROVENANCE_FILE)
    if golden is not None:
        _write_table(out / schema.GOLDEN_FILE, golden, schema.GOLDEN_COLUMNS, lambda r: (r["kind"], r["url"]))
        files.append(schema.GOLDEN_FILE)

    counts = dict.fromkeys(schema.SERVICES, 0)
    counts.update(Counter(r["service"] for r in rows))
    manifest = {
        "schema_version": schema.SCHEMA_VERSION,
        "snapshot_date": snapshot_date or _now().date().isoformat(),
        "files": {
            name: {
                "sha256": schema.sha256_file(out / name),
                "bytes": (out / name).stat().st_size,
                "rows": {
                    schema.SNAPSHOT_FILE: len(rows),
                    schema.PROVENANCE_FILE: len(provenance or []),
                    schema.GOLDEN_FILE: len(golden or []),
                }[name],
            }
            for name in files
        },
        "rows_per_service": dict(sorted(counts.items())),
        "libraries": library_versions(),
    }
    if extra:
        manifest.update(extra)
    schema.write_manifest(out, manifest)
    return manifest
