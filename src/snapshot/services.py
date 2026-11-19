"""Service emulation: turn requests into native responses from a :class:`SnapshotStore`.

Adapted from the EvidenceQuest local snapshot (Polygl0t/EvidenceQuest), reduced to the services the SciFlawBench
tools use: web search, page fetch, Wikipedia and arXiv.

Each method returns `(status, content_type, body)` exactly as the server sends it, so the logic is testable without
HTTP. Two request classes are handled differently:

- exact keys (page URLs, arXiv ids, Wikipedia titles): dictionary lookup on the canonical request key;
- free text (web search, Wikipedia search, arXiv queries): BM25 over the global corpus, ties broken by `doc_id`.

A miss always yields the service's own "nothing found" shape.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from functools import cached_property
from typing import Any

from . import schema
from .bm25 import BM25Index, tokenize
from .canon import (
    canonical_arxiv_id,
    canonical_url,
    canonical_wiki_title,
    classify_url,
    fold,
)

Response = tuple[int, str, str]

_JSON = "application/json; charset=utf-8"
_TEXT = "text/plain; charset=utf-8"
_MARKDOWN = "text/markdown; charset=utf-8"
_ATOM = "application/atom+xml; charset=utf-8"

_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
ET.register_namespace("", _ATOM_NS)
ET.register_namespace("arxiv", _ARXIV_NS)
ET.register_namespace("opensearch", _OPENSEARCH_NS)

# Free-text thresholds: a hit needs a BM25 score above these to be returned.
WEB_MIN_SCORE = 1.0
WIKI_MIN_SCORE = 1.0
ARXIV_MIN_SCORE = 0.5

_HTTP_REASONS = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    410: "Gone",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
}


def wiki_page_url(lang: str, title: str) -> str:
    """The article URL MediaWiki reports (`Irati_(Paran%C3%A1)`: parentheses unescaped)."""
    slug = urllib.parse.quote(title.replace(" ", "_"), safe="/:()!,;@$*'~-._")
    return f"https://{lang}.wikipedia.org/wiki/{slug}"


def http_reason(status: int) -> str:
    return _HTTP_REASONS.get(int(status), "Error")


def _json(data: Any, status: int = 200) -> Response:
    return status, _JSON, json.dumps(data, ensure_ascii=False)


# A stored `markdown` value starting with this marker records that the live
# `visit_webpage` could not convert the page; the rest is the exception message.
RENDER_ERROR_PREFIX = "\x00visit_webpage-error:"


class MarkdownRenderError(Exception):
    """markdownify failed on a page (e.g. RecursionError on deeply nested HTML)."""


def _wiki_result_title(lang: str, title: str) -> str:
    """How search engines title a Wikipedia article."""
    return f"{title} - Wikipedia"


def render_markdown(html: str) -> str:
    """The `visit_webpage` rendering: markdownify, strip, collapse 3+ newlines.

    markdownify recurses once per nesting level, so pages with a few hundred nested (or
    unclosed) tags exceed Python's recursion limit.  The live tool catches that and
    answers `An unexpected error occurred: <message>`; this raises
    :class:`MarkdownRenderError` with the same message so callers can reproduce it.
    """
    from markdownify import markdownify

    try:
        text = markdownify(html).strip()
    except RecursionError as exc:
        raise MarkdownRenderError(str(exc)) from None
    except Exception as exc:  # the live tool reports any conversion failure the same way
        raise MarkdownRenderError(str(exc)) from exc
    return re.sub(r"\n{3,}", "\n\n", text)


def _render_error(message: str) -> Response:
    """What `/web/fetch` answers for a page the live tool cannot convert."""
    return _json(
        {"status": 500, "reason": "Internal Server Error", "unexpected_error": message},
        status=500,
    )


def synth_snippet(text: str, query: str, width: int = 160) -> str:
    """Deterministic snippet: the earliest sentence window with the most query terms."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    terms = set(tokenize(query))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    best_hits, offset, best_offset = -1, 0, 0
    for sentence in sentences:
        hits = len(terms & set(tokenize(sentence)))
        if hits > best_hits:
            best_hits, best_offset = hits, offset
        offset += len(sentence) + 1
    window = text[best_offset : best_offset + width * 2]
    if len(window) <= width:
        return window
    cut = window[:width].rsplit(" ", 1)[0]
    return cut + " ..."


class LocalServices:
    """All service emulators over one store; indexes are built lazily on first use."""

    def __init__(self, store):
        self.store = store
        self.date = store.snapshot_date or "1970-01-01"

    @cached_property
    def web_index(self):
        """BM25 over the pages a search engine would list."""
        index = BM25Index()
        rows: dict[str, dict] = {}
        wiki_web: dict[tuple[str, str], dict] = {}  # (lang, title) -> stored HTML row
        for row in self.store.light_rows(schema.WEB):
            service, key = classify_url(row.get("url") or "")
            if service == schema.WIKIPEDIA:
                lang, title = key.split(":", 1)
                target = self._wiki_resolve(lang, title)
                if target is not None:
                    wiki_web.setdefault((lang, target), row)
                    if int(row.get("status") or 0) != 200:
                        continue
            if int(row.get("status") or 0) != 200:
                continue
            index.add(row["doc_id"], row.get("title") or "", row.get("search_text") or "")
            rows[row["doc_id"]] = row
        for lang in sorted(self.wiki):
            entry = self.wiki[lang]
            for wiki_doc in sorted(entry["rows"]):
                item = entry["rows"][wiki_doc]
                html = wiki_web.get((lang, item["title"]))
                if html is not None and int(html.get("status") or 0) == 200:
                    continue  # already indexed from its own page
                wrow = item["row"]
                doc = html["doc_id"] if html is not None else wiki_doc
                title = _wiki_result_title(lang, item["title"])
                text = wrow.get("search_text") or ""
                index.add(doc, title, text)
                rows[doc] = {
                    "doc_id": doc,
                    "url": wrow.get("url") or wiki_page_url(lang, item["title"]),
                    "title": title,
                    # The engine's own snippet when the build captured one.
                    "snippet": (html or {}).get("snippet") or "",
                    "search_text": text,
                }
        return index.finalize(), rows

    @cached_property
    def wiki(self) -> dict[str, dict]:
        """Per Wikipedia edition: BM25 index, title index, redirect map, rows by doc_id."""
        langs: dict[str, dict] = {}
        for row in self.store.iter_rows(schema.WIKIPEDIA):
            lang = row.get("lang") or row["request_key"].split(":", 1)[0]
            entry = langs.setdefault(
                lang,
                {
                    "index": BM25Index(),
                    "title_index": BM25Index(title_weight=1),
                    "titles": {},
                    "redirects": {},
                    "rows": {},
                },
            )
            body = json.loads(row["body"])
            title = canonical_wiki_title(row["request_key"].split(":", 1)[1])
            entry["titles"][title] = row["doc_id"]
            for alias in body.get("redirects") or []:
                entry["redirects"].setdefault(canonical_wiki_title(alias), title)
            entry["rows"][row["doc_id"]] = {"row": row, "body": body, "title": title}
            entry["index"].add(row["doc_id"], title, row.get("search_text") or "")
            entry["title_index"].add(row["doc_id"], " ".join([title, *(body.get("redirects") or [])]), "")
        for entry in langs.values():
            entry["index"].finalize()
            entry["title_index"].finalize()
        return langs

    @cached_property
    def arxiv(self) -> dict:
        index = BM25Index()
        docs: dict[str, dict] = {}
        for row in self.store.iter_rows(schema.ARXIV):
            entry = _first_atom_entry(row["body"])
            if entry is None:
                continue
            fields = _arxiv_fields(entry)
            fields["entry"] = entry
            fields["key"] = row["request_key"]
            docs[row["doc_id"]] = fields
            index.add(row["doc_id"], fields["ti"], " ".join([fields["abs"], fields["au"]]))
        return {
            "index": index.finalize(),
            "docs": docs,
            "by_key": {d["key"]: i for i, d in docs.items()},
        }

    def web_results(self, query: str, num: int = 10) -> list[dict]:
        index, rows = self.web_index
        hits = index.search(query, limit=max(1, min(int(num or 10), 100)), min_score=WEB_MIN_SCORE)
        results = []
        for position, (doc, _score) in enumerate(hits, 1):
            row = rows[doc]
            snippet = row.get("snippet") or synth_snippet(row.get("search_text") or "", query)
            results.append(
                {
                    "position": position,
                    "title": row.get("title") or row.get("url"),
                    "link": row.get("url"),
                    "snippet": snippet,
                }
            )
        return results

    def serpapi(self, params: dict[str, str]) -> Response:
        query = params.get("q", "")
        results = self.web_results(query, _int(params.get("num"), 10))
        meta = {"status": "Success", "engine": "local-snapshot", "snapshot_date": self.date}
        search_parameters = {k: v for k, v in sorted(params.items()) if k != "api_key"}
        if not results:
            return _json(
                {
                    "search_metadata": meta,
                    "search_parameters": search_parameters,
                    "search_information": {"organic_results_state": "Fully empty"},
                    "error": "Google hasn't returned any results for this query.",
                }
            )
        return _json(
            {
                "search_metadata": meta,
                "search_parameters": search_parameters,
                "organic_results": results,
            }
        )

    def serper(self, params: dict[str, Any]) -> Response:
        query = str(params.get("q", ""))
        results = self.web_results(query, _int(params.get("num"), 10))
        return _json({"searchParameters": {"q": query, "type": "search"}, "organic": results})

    def fetch(self, url: str) -> Response:
        """Return the `visit_webpage` markdown for `url`, or an error status.

        API hosts (arXiv export, the MediaWiki API) are answered by the
        emulators and rendered like any other page; everything else is a stored page.
        """
        service, key = classify_url(url)
        parts = urllib.parse.urlsplit(canonical_url(url))
        query = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
        dynamic: Response | None = None
        if parts.netloc == "export.arxiv.org" and parts.path == "/api/query":
            dynamic = self.arxiv_query(query)
        elif parts.netloc.endswith(".wikipedia.org") and parts.path == "/w/api.php":
            dynamic = self.wikipedia_api(parts.netloc.split(".", 1)[0], query)
        if dynamic is not None:
            status, _ctype, body = dynamic
            if status != 200:
                return _json({"status": status, "reason": http_reason(status)}, status=status)
            try:
                return 200, _MARKDOWN, render_markdown(body)
            except MarkdownRenderError as exc:
                return _render_error(str(exc))

        row = self.store.get(schema.WEB, canonical_url(url))
        if row is None and service == schema.WIKIPEDIA:
            lang, title = key.split(":", 1)
            target = self._wiki_resolve(lang, title)
            if target is not None:
                row = self.store.get(schema.WEB, canonical_url(wiki_page_url(lang, target)))
        if row is None:
            return _json({"status": 404, "reason": "Not Found"}, status=404)
        status = int(row.get("status") or 200)
        if status != 200:
            return _json({"status": status, "reason": http_reason(status)}, status=status)
        markdown = row.get("markdown") or ""
        if markdown.startswith(RENDER_ERROR_PREFIX):
            return _render_error(markdown[len(RENDER_ERROR_PREFIX) :])
        return 200, _MARKDOWN, markdown

    def _wiki_resolve(self, lang: str, title: str) -> str | None:
        entry = self.wiki.get(lang)
        if entry is None:
            return None
        norm = canonical_wiki_title(title)
        if norm in entry["titles"]:
            return norm
        return entry["redirects"].get(norm)

    def wikipedia_api(self, lang: str, params: dict[str, str]) -> Response:
        action = params.get("action", "")
        entry = self.wiki.get(lang)
        if action == "opensearch":
            return _json(self._wiki_opensearch(entry, lang, params))
        if action == "query" and params.get("list") == "search":
            return _json(self._wiki_search(entry, params))
        if action == "query" and params.get("titles") is not None:
            return _json(self._wiki_titles(entry, lang, params))
        return _json({"batchcomplete": ""})

    def _wiki_opensearch(self, entry, lang: str, params) -> list:
        query = params.get("search", "")
        limit = max(1, min(_int(params.get("limit"), 10), 500))
        titles: list[str] = []
        if entry is not None and query.strip():
            prefix = fold(query).strip()
            names = sorted(set(entry["titles"]) | set(entry["redirects"]))
            prefixed = sorted((n for n in names if fold(n).startswith(prefix)), key=lambda n: (len(n), n))
            for name in prefixed:
                target = entry["redirects"].get(name, name)
                if target not in titles:
                    titles.append(target)
            for doc, _score in entry["title_index"].search(query, limit=limit * 2):
                title = entry["rows"][doc]["title"]
                if title not in titles:
                    titles.append(title)
        titles = titles[:limit]
        urls = [wiki_page_url(lang, t) for t in titles]
        return [query, titles, [""] * len(titles), urls]

    def _wiki_search(self, entry, params) -> dict:
        query = params.get("srsearch", "")
        limit = max(1, min(_int(params.get("srlimit"), 10), 500))
        offset = max(0, _int(params.get("sroffset"), 0))
        hits = []
        total = 0
        if entry is not None:
            ranked = entry["index"].search(query, limit=10**6, min_score=WIKI_MIN_SCORE)
            total = len(ranked)
            for doc, _score in ranked[offset : offset + limit]:
                item = entry["rows"][doc]
                info = item["body"].get("info") or {}
                text = (item["body"].get("extracts") or {}).get("extract") or ""
                hits.append(
                    {
                        "ns": info.get("ns", 0),
                        "title": item["title"],
                        "pageid": info.get("pageid"),
                        "size": info.get("length", len(text)),
                        "wordcount": len(text.split()),
                        "snippet": synth_snippet(text, query),
                        "timestamp": info.get("touched", f"{self.date}T00:00:00Z"),
                    }
                )
        out: dict[str, Any] = {"batchcomplete": ""}
        if offset + limit < total:
            out["continue"] = {"sroffset": offset + limit, "continue": "-||"}
        out["query"] = {"searchinfo": {"totalhits": total}, "search": hits}
        return out

    def _wiki_titles(self, entry, lang: str, params) -> dict:
        props = set((params.get("prop") or "").split("|")) - {""}
        follow = params.get("redirects") not in (None, "", "0", "false")
        normalized, redirects, pages = [], [], {}
        missing = 0
        for raw in [t for t in params.get("titles", "").split("|") if t.strip()]:
            norm = canonical_wiki_title(raw)
            if norm != raw:
                normalized.append({"from": raw, "to": norm})
            target = None
            if entry is not None:
                if norm in entry["titles"]:
                    target = norm
                elif follow and norm in entry["redirects"]:
                    target = entry["redirects"][norm]
                    redirects.append({"from": norm, "to": target})
            if target is None:
                missing += 1
                page: dict[str, Any] = {"ns": 0, "title": norm, "missing": ""}
                if "info" in props:
                    page.update(
                        {
                            "contentmodel": "wikitext",
                            "pagelanguage": lang,
                            "pagelanguagehtmlcode": lang,
                            "pagelanguagedir": "ltr",
                        }
                    )
                pages[str(-missing)] = page
                continue
            item = entry["rows"][entry["titles"][target]]
            info = item["body"].get("info") or {}
            page = {"pageid": info.get("pageid"), "ns": info.get("ns", 0), "title": target}
            if "info" in props:
                page.update(info)
            if "extracts" in props:
                ext = item["body"].get("extracts") or {}
                page["extract"] = ext.get("extract", "")
            pages[str(page["pageid"])] = page
        query: dict[str, Any] = {}
        if normalized:
            query["normalized"] = normalized
        if redirects:
            query["redirects"] = redirects
        query["pages"] = pages
        return {"batchcomplete": "", "query": query}

    def arxiv_query(self, params: dict[str, str]) -> Response:
        data = self.arxiv
        start = max(0, _int(params.get("start"), 0))
        max_results = max(0, _int(params.get("max_results"), 10))
        search_query = (params.get("search_query") or "").strip()
        id_list = [canonical_arxiv_id(i) for i in (params.get("id_list") or "").split(",") if i.strip()]

        if id_list:
            ids = [data["by_key"][i] for i in id_list if i in data["by_key"]]
            if search_query:
                keep = _arxiv_match(_arxiv_parse(search_query), data["docs"])
                ids = [i for i in ids if i in keep]
            ordered = ids
        elif search_query:
            candidates = _arxiv_match(_arxiv_parse(search_query), data["docs"])
            sort_by = (params.get("sortBy") or "relevance").lower()
            descending = (params.get("sortOrder") or "descending").lower() != "ascending"
            if sort_by in ("submitteddate", "lastupdateddate"):
                field = "published" if sort_by == "submitteddate" else "updated"
                ordered = sorted(candidates, key=lambda d: (data["docs"][d][field], d), reverse=descending)
            else:
                words = " ".join(v for _f, v in _arxiv_terms(search_query))
                ranked = data["index"].search(words, limit=len(candidates) or 1, candidates=candidates)
                ordered = [d for d, _ in ranked]
                ordered += sorted(candidates - set(ordered))
                if not descending:
                    ordered.reverse()
        else:
            ordered = []

        page = ordered[start : start + max_results]
        feed = ET.Element(f"{{{_ATOM_NS}}}feed")
        ET.SubElement(feed, f"{{{_ATOM_NS}}}id").text = "https://arxiv.org/api/local"
        ET.SubElement(feed, f"{{{_ATOM_NS}}}title").text = "arXiv Query: " + urllib.parse.urlencode(
            sorted(params.items())
        )
        ET.SubElement(feed, f"{{{_ATOM_NS}}}updated").text = f"{self.date}T00:00:00Z"
        ET.SubElement(feed, f"{{{_OPENSEARCH_NS}}}totalResults").text = str(len(ordered))
        ET.SubElement(feed, f"{{{_OPENSEARCH_NS}}}startIndex").text = str(start)
        ET.SubElement(feed, f"{{{_OPENSEARCH_NS}}}itemsPerPage").text = str(max_results)
        for doc in page:
            feed.append(data["docs"][doc]["entry"])
        body = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(feed, encoding="unicode")
        return 200, _ATOM, body

    def health(self) -> dict[str, Any]:
        out = {"status": "ok"}
        out.update(self.store.summary())
        return out


_ARXIV_FIELDS = ("ti", "au", "abs", "co", "jr", "cat", "rn", "id", "all", "submitteddate", "lastupdateddate")
# `submittedDate:[202001010000 TO 299912312359]` is one token, as the range holds spaces.
_ARXIV_RANGE_FIELDS = {"submitteddate": "published", "lastupdateddate": "updated"}
_ARXIV_TOKEN = re.compile(r'\(|\)|"[^"]*"|[A-Za-z]+:\[[^\]]*\]|[^\s()"]+(?:"[^"]*")?')
_ARXIV_RANGE = re.compile(r"^\[\s*(\d*)\s+TO\s+(\d*)\s*\]$", re.IGNORECASE)


def _text(elem, path: str) -> str:
    found = elem.find(path, {"a": _ATOM_NS, "arxiv": _ARXIV_NS})
    return (found.text or "").strip() if found is not None and found.text else ""


def _first_atom_entry(body: str):
    try:
        root = ET.fromstring(body.encode("utf-8"))
    except ET.ParseError:
        return None
    if root.tag == f"{{{_ATOM_NS}}}entry":
        return root
    return root.find(f"{{{_ATOM_NS}}}entry")


def _arxiv_fields(entry) -> dict[str, str]:
    ns = {"a": _ATOM_NS, "arxiv": _ARXIV_NS}
    authors = [(a.findtext("a:name", "", ns) or "").strip() for a in entry.findall("a:author", ns)]
    cats = [c.get("term", "") for c in entry.findall("a:category", ns)]
    ident = _text(entry, "a:id")
    fields = {
        "ti": re.sub(r"\s+", " ", _text(entry, "a:title")),
        "au": "; ".join(authors),
        "abs": re.sub(r"\s+", " ", _text(entry, "a:summary")),
        "co": _text(entry, "arxiv:comment"),
        "jr": _text(entry, "arxiv:journal_ref"),
        "cat": " ".join(cats),
        "rn": "",
        "id": canonical_arxiv_id(ident.rsplit("/abs/", 1)[-1]) if ident else "",
        "published": _text(entry, "a:published"),
        "updated": _text(entry, "a:updated"),
    }
    fields["all"] = " ".join(fields[k] for k in ("ti", "au", "abs", "co", "jr", "cat", "id"))
    return fields


def _arxiv_parse(query: str):
    """Parse an arXiv `search_query` into a small boolean tree.

    Grammar: expr := term ((AND|OR|ANDNOT)? term)*; term := '(' expr ')' | [field:]value.
    Adjacent terms without an operator are OR-ed (they are still ranked by BM25).
    """
    tokens = _ARXIV_TOKEN.findall(query)
    pos = 0

    def term():
        nonlocal pos
        if pos >= len(tokens):
            return None
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            node = expr()
            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1
            return node
        field, value = "all", tok
        m = re.match(r"^([a-z]+):(.*)$", tok, re.IGNORECASE)
        if m and m.group(1).lower() in _ARXIV_FIELDS:
            field, value = m.group(1).lower(), m.group(2)
            if value == "" and pos < len(tokens):
                value = tokens[pos]
                pos += 1
        return ("leaf", field, value.strip('"'), value.startswith('"'))

    def expr():
        nonlocal pos
        left = term()
        while pos < len(tokens) and tokens[pos] != ")":
            op = tokens[pos].upper()
            if op in ("AND", "OR", "ANDNOT"):
                pos += 1
            else:
                op = "OR"
            right = term()
            if right is None:
                break
            left = (op, left, right) if left is not None else right
        return left

    return expr()


def _arxiv_terms(query: str) -> list[tuple[str, str]]:
    out = []

    def walk(node):
        if node is None:
            return
        if node[0] == "leaf":
            if node[1] not in _ARXIV_RANGE_FIELDS:  # a date range filters; it adds nothing to the ranking words
                out.append((node[1], node[2]))
        elif node[0] != "ANDNOT":
            walk(node[1])
            walk(node[2])
        else:
            walk(node[1])

    walk(_arxiv_parse(query))
    return out


def _arxiv_in_range(value: str, field: str, docs: dict[str, dict]) -> set[str]:
    """Docs whose date (`published` or `updated`) lies in `[YYYYMMDDHHMM TO YYYYMMDDHHMM]` (bounds may be shorter)."""
    m = _ARXIV_RANGE.match(value.strip())
    if m is None:
        return set()
    low, high = m.group(1).ljust(12, "0"), m.group(2).ljust(12, "9")
    hits = set()
    for doc, fields in docs.items():
        stamp = re.sub(r"\D", "", fields.get(field, ""))[:12].ljust(12, "0")
        if low <= stamp <= high:
            hits.add(doc)
    return hits


def _arxiv_match(node, docs: dict[str, dict]) -> set[str]:
    if node is None:
        return set()
    if node[0] == "leaf":
        _, field, value, phrase = node
        if field in _ARXIV_RANGE_FIELDS:
            return _arxiv_in_range(value, _ARXIV_RANGE_FIELDS[field], docs)
        wanted = tokenize(value, keep_stopwords=True)
        if not wanted:
            return set()
        hits = set()
        for doc, fields in docs.items():
            if field == "cat":
                cats = fold(fields["cat"]).split()
                v = fold(value)
                ok = any(c == v or (v.endswith("*") and c.startswith(v[:-1])) for c in cats)
            elif field == "id":
                ok = fields["id"] == canonical_arxiv_id(value)
            else:
                have = tokenize(fields.get(field, ""), keep_stopwords=True)
                if phrase:
                    n = len(wanted)
                    ok = any(have[i : i + n] == wanted for i in range(len(have) - n + 1))
                else:
                    ok = all(w in have for w in wanted)
            if ok:
                hits.add(doc)
        return hits
    op, left, right = node
    a, b = _arxiv_match(left, docs), _arxiv_match(right, docs)
    if op == "AND":
        return a & b
    if op == "ANDNOT":
        return a - b
    return a | b


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
