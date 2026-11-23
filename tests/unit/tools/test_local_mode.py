import pytest
import requests

from tools.definitions import tool_registry
from tools.localmode import LocalSearchTool, LocalVisitWebpageTool
from tools.searchtools import ArxivSearchTool, ImprovedWikipediaSearchTool
from tools.webpage import PublicVisitWebpageTool

BASE = "http://127.0.0.1:8123"


class _Reply:
    def __init__(self, payload=None, text="", status_code=200, reason="OK"):
        self._payload, self.text, self.status_code, self.reason = payload, text, status_code, reason

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


@pytest.fixture
def get(monkeypatch):
    """Replace `requests.get` in the local-mode module: returns the calls made and a list of replies to fill."""
    calls, replies = [], []

    def fake_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params})
        return replies.pop(0)

    monkeypatch.setattr("tools.localmode.requests.get", fake_get)
    return calls, replies


def test_search_goes_to_the_local_route_and_keeps_the_live_format(get):
    calls, replies = get
    replies.append(_Reply({"organic": [{"title": "GW190521", "link": "https://x.org/a", "snippet": "A merger."}]}))
    out = LocalSearchTool(BASE, max_results=5).forward("gw190521")
    assert out == "## Search Results\n\n[GW190521](https://x.org/a)\nA merger."
    assert calls == [{"url": f"{BASE}/serper/search", "params": {"q": "gw190521", "num": 5}}]


def test_search_without_results_raises_the_live_error(get):
    _, replies = get
    replies.append(_Reply({"organic": []}))
    with pytest.raises(Exception, match="No results found"):
        LocalSearchTool(BASE).forward("nothing")


def test_search_tool_looks_like_the_live_one():
    tool = LocalSearchTool(BASE)
    assert tool.name == "web_search" and "query" in tool.inputs


def test_visit_webpage_asks_the_fetch_route_and_returns_markdown(get):
    calls, replies = get
    replies.append(_Reply(text="\n# Title\n\n\n\n\nbody\n"))
    out = LocalVisitWebpageTool(BASE).forward("https://x.org/a")
    assert out == "# Title\n\nbody"
    assert calls == [{"url": f"{BASE}/web/fetch", "params": {"url": "https://x.org/a"}}]


def test_a_page_missing_from_the_snapshot_answers_like_a_live_404(get):
    _, replies = get
    replies.append(_Reply({}, status_code=404, reason="Not Found"))
    out = LocalVisitWebpageTool(BASE).forward("https://x.org/missing")
    assert out == "Error fetching the webpage: 404 Client Error: Not Found for url: https://x.org/missing"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x.org/f", "http:///x"])
def test_visit_webpage_refuses_other_schemes_without_a_request(get, url):
    calls, _ = get
    assert LocalVisitWebpageTool(BASE).forward(url).startswith("Error fetching the webpage:")
    assert calls == []


def test_wikipedia_uses_the_local_api(monkeypatch):
    tool = ImprovedWikipediaSearchTool("me@x.org", base_url=BASE)
    assert tool.SEARCH_API_URL == BASE + "/wikipedia/{language}/w/api.php"
    assert tool.wiki._build_url("en") == BASE + "/wikipedia/en/w/api.php"


def test_arxiv_uses_the_local_route_and_no_wait():
    tool = ArxivSearchTool("me@x.org", base_url=BASE)
    assert tool.BASE_URL == BASE + "/arxiv/api/query"
    assert tool.MIN_INTERVAL_S == 0.0


def test_a_trailing_slash_in_base_url_is_ignored():
    assert ArxivSearchTool("me@x.org", base_url=BASE + "/").BASE_URL == BASE + "/arxiv/api/query"


def test_registry_builds_local_tools_with_base_url():
    assert isinstance(tool_registry.get("web_search")(watcher=None, base_url=BASE)._wrapped, LocalSearchTool)
    assert isinstance(tool_registry.get("visit_webpage")(watcher=None, base_url=BASE)._wrapped, LocalVisitWebpageTool)
    assert tool_registry.get("arxiv_search")(watcher=None, base_url=BASE)._wrapped.BASE_URL.startswith(BASE)
    assert tool_registry.get("wikipedia_search")(watcher=None, base_url=BASE)._wrapped.base_url == BASE


def test_registry_defaults_are_unchanged():
    assert isinstance(tool_registry.get("visit_webpage")(watcher=None)._wrapped, PublicVisitWebpageTool)
    assert tool_registry.get("arxiv_search")(watcher=None)._wrapped.BASE_URL == "https://export.arxiv.org/api/query"
    assert tool_registry.get("wikipedia_search")(watcher=None)._wrapped.base_url is None


def test_the_benchmarks_own_repository_is_refused_before_any_request(get):
    calls, _ = get
    out = LocalVisitWebpageTool(BASE).forward("https://github.com/caisa-lab/SciFlawBench/blob/main/x")
    assert out == "Error fetching the webpage: 'https://github.com/caisa-lab/SciFlawBench/blob/main/x' is not available"
    assert "not available" in LocalVisitWebpageTool(BASE).forward("https://x.org/sciflawbench%2Ftasks")
    assert calls == []


def test_search_results_from_the_benchmarks_repository_are_dropped(get):
    _, replies = get
    replies.append(
        _Reply(
            {
                "organic": [
                    {"title": "tasks", "link": "https://github.com/caisa-lab/SciFlawBench", "snippet": "answers"},
                    {"title": "GW", "link": "https://x.org/a", "snippet": "A merger."},
                ]
            }
        )
    )
    out = LocalSearchTool(BASE).forward("gw")
    assert "SciFlawBench" not in out and "(https://x.org/a)" in out
