"""Local mode: the web tools answer from a local server (a frozen snapshot) instead of the live internet.

Every network tool takes an optional `base_url`. With `base_url=None` (the default) nothing changes. With the address of
a local server, a tool sends the same kind of request to that server, and its name, inputs, output format and error
messages stay the same, so an agent cannot tell the difference and a run can be repeated with the same pages.

The server's routes (the same names as in the snapshot server):

| Route                           | Used by                                  |
|---------------------------------|------------------------------------------|
| GET /serper/search              | `web_search` (`LocalSearchTool`)         |
| GET /web/fetch?url=             | `visit_webpage` (`LocalVisitWebpageTool`) |
| GET /wikipedia/{lang}/w/api.php | `wikipedia_search`                       |
| GET /arxiv/api/query            | `arxiv_search`                           |
"""

import re
from collections.abc import Sequence
from urllib.parse import unquote, urlsplit

import requests
from smolagents import DuckDuckGoSearchTool, Tool, VisitWebpageTool

from tools.webpage import BLOCKED_URLS

ROUTE_SEARCH = "/serper/search"
ROUTE_WEB_FETCH = "/web/fetch"
ROUTE_WIKIPEDIA = "/wikipedia/{language}/w/api.php"
ROUTE_ARXIV = "/arxiv/api/query"

TIMEOUT_S = 30


def local_url(base_url: str, route: str) -> str:
    """The address of `route` on the local server at `base_url`."""
    return base_url.rstrip("/") + route


def _blocked(url: str, patterns: Sequence[str]) -> bool:
    """Whether the URL holds one of the blocked patterns, as the live tools decide it (case and %-escapes ignored)."""
    return any(pattern in unquote(url).lower() for pattern in patterns)


class LocalSearchTool(Tool):
    """`web_search` answered by the local server: the same name, input and output format as the DuckDuckGo search.

    Results come from the server's own ranking over the snapshot, so the same query always gives the same results. A
    search without results raises the same error as the live search, so the agent sees what it would see live.
    """

    name = "web_search"
    description = DuckDuckGoSearchTool.description
    inputs = DuckDuckGoSearchTool.inputs
    output_type = "string"

    def __init__(self, base_url: str, max_results: int = 8, blocked_urls: Sequence[str] = BLOCKED_URLS):
        super().__init__()
        self.base_url = base_url
        self.max_results = max_results
        self.blocked_urls = [pattern.lower() for pattern in blocked_urls if pattern]

    def forward(self, query: str) -> str:
        response = requests.get(
            local_url(self.base_url, ROUTE_SEARCH), params={"q": query, "num": self.max_results}, timeout=TIMEOUT_S
        )
        response.raise_for_status()
        results = (response.json() or {}).get("organic") or []
        # Like the live searches, drop results whose URL holds a blocked pattern (the benchmark's own repository).
        results = [r for r in results if not _blocked(r.get("link", ""), self.blocked_urls)]
        if not results:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        pages = [f"[{r.get('title', '')}]({r.get('link', '')})\n{r.get('snippet', '')}" for r in results]
        return "## Search Results\n\n" + "\n\n".join(pages)


class LocalVisitWebpageTool(VisitWebpageTool):
    """`visit_webpage` answered by the local server, which holds each page already converted to Markdown.

    Same name, input, truncation and error messages as the live tool. A page that is not in the snapshot answers like a
    live 404. The address checks of the live tool (public addresses only) are not needed: nothing here leaves the
    machine, and the server only looks the URL up.
    """

    def __init__(self, base_url: str, max_output_length: int = 40000, blocked_urls: Sequence[str] = BLOCKED_URLS):
        super().__init__(max_output_length=max_output_length)
        self.base_url = base_url
        self.blocked_urls = [pattern.lower() for pattern in blocked_urls if pattern]

    def forward(self, url: str) -> str:
        if _blocked(url, self.blocked_urls):  # the live tool's refusal, before any request, without a reason
            return f"Error fetching the webpage: {url!r} is not available"
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return f"Error fetching the webpage: only http and https URLs can be visited, not {url!r}"
        if not parts.hostname:
            return f"Error fetching the webpage: {url!r} has no host"
        try:
            response = requests.get(local_url(self.base_url, ROUTE_WEB_FETCH), params={"url": url}, timeout=TIMEOUT_S)
        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except requests.exceptions.RequestException as exc:
            return f"Error fetching the webpage: {exc}"
        if response.status_code != 200:
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            if "unexpected_error" in payload:  # the live tool could not convert this page to Markdown
                return f"An unexpected error occurred: {payload['unexpected_error']}"
            status = response.status_code
            kind = "Client" if 400 <= status < 500 else "Server"
            reason = payload.get("reason") or response.reason
            # The text `requests` gives for the live URL.
            return f"Error fetching the webpage: {status} {kind} Error: {reason} for url: {url}"
        markdown = re.sub(r"\n{3,}", "\n\n", response.text.strip())
        return self._truncate_content(markdown, self.max_output_length)


def local_wikipedia(base_url: str, user_agent: str, language: str, extract_format):
    """A `wikipediaapi.Wikipedia` whose MediaWiki API address is the local server's."""
    import wikipediaapi

    class _LocalWikipedia(wikipediaapi.Wikipedia):
        def _build_url(self, language: str) -> str:  # hook of wikipedia-api
            return local_url(base_url, ROUTE_WIKIPEDIA.format(language=language))

    return _LocalWikipedia(user_agent=user_agent, language=language, extract_format=extract_format)
