"""Request canonicalization to make one stable `request_key` per servable item."""

from __future__ import annotations

import re
import unicodedata
import urllib.parse

from . import schema

# Query parameters that never change what a page shows.
_TRACKING_PARAMS = re.compile(
    r"^(utm_[a-z]+|fbclid|gclid|dclid|msclkid|mc_cid|mc_eid|_ga|_gl|yclid|igshid|ref_src)$",
    re.IGNORECASE,
)
# Characters left unescaped in a canonical path (RFC 3986 pchar minus '%').
_PATH_SAFE = "/:@!$&'()*+,;=-._~"
_QUERY_SAFE = ":@!$'()*+,;/?-._~"

_WIKI_HOST = re.compile(r"^([a-z\-]+)\.(?:m\.)?wikipedia\.org$")
_ARXIV_PREFIX = re.compile(r"^(?:arxiv:)", re.IGNORECASE)
_ARXIV_URL = re.compile(r"^https?://(?:export\.|www\.)?arxiv\.org/(?:abs|pdf|html)/(.+?)(?:\.pdf)?/?$", re.IGNORECASE)
_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)


def fold(text: str) -> str:
    """Lowercase and strip combining accents (the tokeniser's normal form)."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def canonical_wiki_title(title: str) -> str:
    """MediaWiki title normalisation: underscores to spaces, single spaces, capital first."""
    text = urllib.parse.unquote(str(title)).replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:] if text else text


def wiki_key(lang: str, title: str) -> str:
    return f"{(lang or 'en').lower()}:{canonical_wiki_title(title)}"


def canonical_arxiv_id(value: str) -> str:
    """arXiv id without `arxiv:` prefix, URL or version suffix (`1205.2469v2` -> `1205.2469`)."""
    text = str(value).strip()
    match = _ARXIV_URL.match(text)
    if match:
        text = match.group(1)
    text = _ARXIV_PREFIX.sub("", text)
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    return _ARXIV_VERSION.sub("", text).strip("/")


def canonical_url(url: str) -> str:
    """Normalise a web URL into its lookup key."""
    text = str(url).strip()
    if not re.match(r"^[a-z][a-z0-9+.\-]*://", text, re.IGNORECASE):
        text = "https://" + text.lstrip("/")
    parts = urllib.parse.urlsplit(text)
    host = (parts.hostname or "").lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    wiki = _WIKI_HOST.match(host)
    if wiki:
        host = f"{wiki.group(1)}.wikipedia.org"
    port = parts.port
    netloc = host if port in (None, 80, 443) else f"{host}:{port}"

    path = urllib.parse.unquote(parts.path or "/")
    if wiki and path.startswith("/wiki/"):
        path = "/wiki/" + canonical_wiki_title(path[len("/wiki/") :]).replace(" ", "_")
    path = urllib.parse.quote(path, safe=_PATH_SAFE)
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    pairs = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if not _TRACKING_PARAMS.match(key)
    ]
    pairs.sort()
    query = urllib.parse.urlencode(pairs, quote_via=urllib.parse.quote, safe=_QUERY_SAFE)
    return urllib.parse.urlunsplit(("https", netloc, path, query, ""))


def classify_url(url: str) -> tuple[str, str]:
    """Map a source URL onto `(service, request_key)`."""
    canon = canonical_url(url)
    parts = urllib.parse.urlsplit(canon)
    host = parts.netloc
    path = urllib.parse.unquote(parts.path)

    wiki = _WIKI_HOST.match(host)
    if wiki and path.startswith("/wiki/"):
        return schema.WIKIPEDIA, wiki_key(wiki.group(1), path[len("/wiki/") :])
    if host in ("arxiv.org", "export.arxiv.org") and re.match(r"^/(abs|pdf)/", path):
        return schema.ARXIV, canonical_arxiv_id(path.split("/", 2)[2])
    return schema.WEB, canon
