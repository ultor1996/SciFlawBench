"""The step 1 tools against the real local server, with every non-loopback connection refused."""

import socket

import pytest

from snapshot.runtime import block_outbound_network, wait_for_server
from tools.definitions import tool_registry


@pytest.fixture(autouse=True)
def no_internet():
    with block_outbound_network():
        yield


def _tool(name, server, **options):
    return tool_registry.get(name)(watcher=None, base_url=server.url, **options)._wrapped


def test_health_reports_the_snapshot(server):
    health = wait_for_server(server.url, timeout=5)
    assert health["status"] == "ok" and health["snapshot_date"] == "2026-11-01"


def test_web_search_and_visit_webpage(server):
    results = _tool("web_search", server).forward("evolving dark energy")
    assert results.startswith("## Search Results") and "(https://example.org/desi)" in results
    page = _tool("visit_webpage", server).forward("https://example.org/gw")
    assert "heavy binary black hole merger" in page
    assert _tool("visit_webpage", server).forward("https://example.org/missing") == (
        "Error fetching the webpage: 404 Client Error: Not Found for url: https://example.org/missing"
    )
    with pytest.raises(Exception, match="No results found"):
        _tool("web_search", server).forward("zebra giraffe")


def test_wikipedia_search(server):
    out = _tool("wikipedia_search", server, operator="me@x.org").forward("Dark energy")
    assert "Dark energy is an unknown form of energy" in out


def test_arxiv_search_with_a_year_filter(server):
    tool = _tool("arxiv_search", server, operator="me@x.org")
    out = tool.forward("DESI results", "relevance", 2020)
    assert "id: 2503.14738" in out and "Old lensing" not in out
    assert "id: 1803.07687" in tool.forward("1803.07687", "relevance", 1991)
    assert "id: 1803.07687" in tool.forward("lensing", "oldest", 1991)


def test_a_live_request_is_really_refused(server):
    with pytest.raises((ConnectionRefusedError, socket.gaierror), match="blocked"):
        socket.create_connection(("93.184.215.14", 80), timeout=2)
