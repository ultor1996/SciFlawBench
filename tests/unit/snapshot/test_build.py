"""The snapshot builder: inventory from a task file, the cached fetcher, and a build that is then served and read."""

import json

import pytest
import requests

from snapshot import build, schema
from snapshot.runtime import BackgroundServer, block_outbound_network
from snapshot.store import SnapshotStore
from tools.definitions import tool_registry

from .conftest import ATOM, PAGE

TASKS = [
    {
        "task_id": "T1",
        "task": "Use https://example.org/data.html (see also https://example.org/table.txt.) to answer.",
        "sources": [
            {"url": "https://arxiv.org/abs/2503.14738v2", "note": "DESI"},
            {"url": "https://en.wikipedia.org/wiki/Dark_energy", "note": "wiki"},
            {"url": "https://example.org/paper", "note": "page"},
        ],
        "validators": [{"name": "judge:rubric", "kwargs": {"rubric": "see https://example.org/key (the key)"}}],
    },
    {"task_id": "T2", "task": "No links here.", "sources": [{"url": "https://example.org/paper"}]},
    {"task_id": "T3", "task": "A source that is gone.", "sources": [{"url": "https://example.org/gone"}]},
]


class FakeFetcher:
    """Answers like the live services, from a dict, and counts the requests."""

    def __init__(self):
        self.calls = []
        self.skip_hosts = set()

    def get(self, url, params=None, *, api=True):
        self.calls.append((url, params))
        stamp = "2026-11-01T00:00:00+00:00"
        if url == build.ARXIV_API:
            feed = ATOM.format(
                id=params["id_list"],
                published="2025-03-18T10:00:00Z",
                title="DESI DR2 results",
                summary="Dark energy.",
                author="A. Author",
            )
            return build.Fetched(url, 200, "application/atom+xml", feed, stamp)
        if url == "https://en.wikipedia.org/w/api.php":
            page = {
                "pageid": 7,
                "ns": 0,
                "title": "Dark energy",
                "fullurl": "https://en.wikipedia.org/wiki/Dark_energy",
            }
            if params.get("prop") == "extracts":
                page["extract"] = "Dark energy is an unknown form of energy."
            if params.get("prop") == "redirects":
                page["redirects"] = [{"title": "Quintessence (physics)"}]
            return build.Fetched(url, 200, "application/json", json.dumps({"query": {"pages": {"7": page}}}), stamp)
        if url.endswith("/gone"):
            return build.Fetched(url, 404, "text/html", "<p>gone</p>", stamp)
        if url.endswith("/key"):
            raise build.FetchError("host unreachable")
        title = url.rsplit("/", 1)[-1]
        return build.Fetched(url, 200, "text/html", PAGE.format(title=title, text=f"text of {title} page"), stamp)


@pytest.fixture
def tasks_file(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in TASKS) + "\n", encoding="utf-8")
    return path


def test_the_inventory_lists_each_request_once_with_its_tasks():
    needs = {(n.role, n.service, n.request_key): n for n in build.build_inventory(TASKS)}
    assert needs[("gold", "arxiv", "2503.14738")].task_ids == {"T1"}
    assert needs[("gold", "wikipedia", "en:Dark energy")].url.endswith("Dark_energy")
    assert needs[("gold", "web", "https://example.org/paper")].task_ids == {"T1", "T2"}
    assert needs[("prompt", "web", "https://example.org/data.html")].task_ids == {"T1"}
    assert (
        "prompt",
        "web",
        "https://example.org/table.txt",
    ) in needs  # the full stop of the sentence is not part of it
    assert needs[("gold", "web", "https://example.org/key")].note == "validators"


def test_urls_in_text_drop_sentence_punctuation():
    assert build.urls_in_text("see (https://a.org/x), and https://b.org/y_(z).") == [
        "https://a.org/x",
        "https://b.org/y_(z)",
    ]


def test_a_build_stores_the_sources_and_reports_the_misses(tasks_file, tmp_path):
    out = tmp_path / "snap"
    manifest = build.build(tasks_file, out, FakeFetcher(), snapshot_date="2026-11-01")
    assert manifest["rows_per_service"] == {"arxiv": 1, "web": 6, "wikipedia": 1}
    assert manifest["tasks"] == 3 and manifest["tasks_sha256"]
    store = SnapshotStore(out)
    assert store.get("web", "https://example.org/gone")["status"] == 404  # stored: live it answers 404 as well
    review = (out / "build_review.md").read_text()
    assert "https://example.org/gone" in review and "HTTP 404" in review
    assert "https://example.org/key" in review and "host unreachable" in review


def test_provenance_records_role_and_tasks(tasks_file, tmp_path):
    import pyarrow.parquet as pq

    build.build(tasks_file, tmp_path / "snap", FakeFetcher(), snapshot_date="2026-11-01")
    table = pq.read_table(tmp_path / "snap" / schema.PROVENANCE_FILE).to_pylist()
    by_url = {(r["source_url"], r["role"]): r for r in table}
    assert by_url[("https://example.org/paper", "gold")]["task_ids"] == ["T1", "T2"]
    assert by_url[("https://example.org/data.html", "prompt")]["task_ids"] == ["T1"]


def test_the_same_inputs_build_the_same_snapshot(tasks_file, tmp_path):
    build.build(tasks_file, tmp_path / "a", FakeFetcher(), snapshot_date="2026-11-01")
    build.build(tasks_file, tmp_path / "b", FakeFetcher(), snapshot_date="2026-11-01")
    # The FakeFetcher stamps every answer with the same time, so the files must match byte for byte.
    assert schema.sha256_file(tmp_path / "a" / schema.SNAPSHOT_FILE) == schema.sha256_file(
        tmp_path / "b" / schema.SNAPSHOT_FILE
    )


def test_a_built_snapshot_is_served_to_the_tools_without_any_network(tasks_file, tmp_path):
    build.build(tasks_file, tmp_path / "snap", FakeFetcher(), snapshot_date="2026-11-01")
    with BackgroundServer(tmp_path / "snap") as server, block_outbound_network():
        visit = tool_registry.get("visit_webpage")(watcher=None, base_url=server.url)._wrapped
        assert "text of paper page" in visit.forward("https://example.org/paper")
        assert "text of Dark" in visit.forward("https://en.wikipedia.org/wiki/Dark_energy")
        wiki = tool_registry.get("wikipedia_search")(watcher=None, base_url=server.url, operator="me@x.org")._wrapped
        assert "Dark energy is an unknown form of energy" in wiki.forward("Dark energy")
        arxiv = tool_registry.get("arxiv_search")(watcher=None, base_url=server.url, operator="me@x.org")._wrapped
        assert "id: 2503.14738" in arxiv.forward("2503.14738", "relevance", 1991)
        search = tool_registry.get("web_search")(watcher=None, base_url=server.url)._wrapped
        assert "example.org/paper" in search.forward("text of paper page")


# the fetcher


class _Reply:
    def __init__(self, status=200, text="body"):
        self.status_code, self.text, self.url = status, text, "https://example.org/final"
        self.headers = {"Content-Type": "text/html"}


class _Session:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _fetcher(tmp_path, replies, **kwargs):
    fetcher = build.Fetcher(tmp_path / "cache", "me@x.org", interval=0, **kwargs)
    fetcher._session = _Session(replies)
    return fetcher


def test_a_response_is_cached_and_then_served_offline(tmp_path):
    fetcher = _fetcher(tmp_path, [_Reply(text="first")])
    assert fetcher.get("https://example.org/a").text == "first"
    assert fetcher.get("https://example.org/a").text == "first"
    assert fetcher._session.calls == 1 and fetcher.stats["cache_hits"] == 1
    offline = build.Fetcher(tmp_path / "cache", "me@x.org", offline=True)
    assert offline.get("https://example.org/a").text == "first"
    with pytest.raises(build.OfflineCacheMiss):
        offline.get("https://example.org/b")


def test_busy_servers_are_asked_again(tmp_path, monkeypatch):
    monkeypatch.setattr("snapshot.build.time.sleep", lambda s: None)
    fetcher = _fetcher(tmp_path, [_Reply(503), _Reply(200, "ok")])
    assert fetcher.get("https://example.org/a").text == "ok"


def test_a_host_that_does_not_answer_is_given_up(tmp_path, monkeypatch):
    monkeypatch.setattr("snapshot.build.time.sleep", lambda s: None)
    fetcher = _fetcher(tmp_path, [requests.ConnectionError("down")] * 3, down_after=3)
    with pytest.raises(build.HostUnavailable):
        fetcher.get("https://example.org/a")
    with pytest.raises(build.HostUnavailable, match="is down"):
        fetcher.get("https://example.org/b")
    assert fetcher._session.calls == 3  # the second request did not even try


def test_a_skipped_host_is_not_contacted(tmp_path):
    fetcher = _fetcher(tmp_path, [], skip_hosts=["example.org"])
    with pytest.raises(build.HostUnavailable, match="skipped"):
        fetcher.get("https://example.org/a")
