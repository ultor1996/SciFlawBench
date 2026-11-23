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
        self.headers = headers
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


def test_cached_errors_are_asked_again_only_on_request(tmp_path):
    fetcher = _fetcher(tmp_path, [_Reply(503, "down")], retries=0)
    assert fetcher.get("https://example.org/a").status == 503
    assert fetcher.get("https://example.org/a").status == 503  # still the cached error
    again = build.Fetcher(tmp_path / "cache", "me@x.org", interval=0, retry_errors=True)
    again._session = _Session([_Reply(200, "back")])
    assert again.get("https://example.org/a").text == "back"
    assert again.get("https://example.org/a").text == "back"  # a good answer is cached for good
    assert again._session.calls == 1
    offline = build.Fetcher(tmp_path / "cache", "me@x.org", offline=True, retry_errors=True)
    assert offline.get("https://example.org/a").text == "back"


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


# check first, then build from what worked


def _checked(tasks_file, tmp_path):
    results = build.check(tasks_file, FakeFetcher())
    build.write_check(tmp_path / "snap", results, tasks_file)
    return results, tmp_path / "snap" / build.CHECK_RESULTS_FILE


def test_the_check_says_which_sources_failed_and_why(tasks_file, tmp_path):
    results, _ = _checked(tasks_file, tmp_path)
    status = {r["url"]: (r["status"], r["reason"]) for r in results}
    assert status["https://example.org/paper"] == ("ok", "")
    assert status["https://arxiv.org/abs/2503.14738v2"][0] == "ok"
    assert status["https://example.org/gone"] == ("failed", "HTTP 404")
    assert status["https://example.org/key"] == ("failed", "host unreachable")
    report = (tmp_path / "snap" / "check_report.md").read_text()
    assert "2 failed" in report and "https://example.org/gone" in report and "--accept" in report


def test_the_check_stores_no_snapshot(tasks_file, tmp_path):
    _checked(tasks_file, tmp_path)
    assert not (tmp_path / "snap" / schema.SNAPSHOT_FILE).exists()


def test_a_build_after_the_check_leaves_out_what_failed(tasks_file, tmp_path):
    _, check_file = _checked(tasks_file, tmp_path)
    manifest = build.build(
        tasks_file, tmp_path / "snap", FakeFetcher(), snapshot_date="2026-11-01", skip=build.failed_to_skip(check_file)
    )
    assert manifest["requests_skipped"] == 2 and manifest["requests_missed"] == 0
    store = SnapshotStore(tmp_path / "snap")
    assert not store.has("web", "https://example.org/gone")
    assert store.has("web", "https://example.org/paper")
    review = (tmp_path / "snap" / "build_review.md").read_text()
    assert "Left out" in review and "https://example.org/gone" in review


def test_an_accepted_failure_is_kept_as_the_error_it_is(tasks_file, tmp_path):
    _, check_file = _checked(tasks_file, tmp_path)
    accept = tmp_path / "accept.txt"
    accept.write_text("# this page is gone for good\nhttps://www.example.org/gone?utm_source=x\n", encoding="utf-8")
    skip = build.failed_to_skip(check_file, accept)
    assert skip == {("web", "https://example.org/key", "gold")}
    build.build(tasks_file, tmp_path / "snap", FakeFetcher(), snapshot_date="2026-11-01", skip=skip)
    assert SnapshotStore(tmp_path / "snap").get("web", "https://example.org/gone")["status"] == 404


# pages that sites refuse to scripts


def test_a_browser_user_agent_is_sent_for_pages_only_when_given(tmp_path):
    plain = _fetcher(tmp_path / "a", [_Reply(), _Reply()])
    plain.get("https://example.org/p", api=False)
    assert "User-Agent" not in plain._session.headers  # like the live tool: requests' own
    browser = _fetcher(tmp_path / "b", [_Reply(), _Reply()], page_user_agent="Mozilla/5.0 test")
    browser.get("https://example.org/p", api=False)
    assert browser._session.headers == {"User-Agent": "Mozilla/5.0 test"}
    browser.get("https://example.org/api", api=True)
    assert "SciFlawBench-snapshot-builder" in browser._session.headers["User-Agent"]


def test_a_page_saved_from_the_browser_replaces_a_cached_error(tmp_path):
    fetcher = _fetcher(tmp_path, [_Reply(403, "blocked")], retries=0)
    assert fetcher.get("https://example.org/a").status == 403
    fetcher.put("https://example.org/a", "<h1>The real page</h1>")
    again = build.Fetcher(tmp_path / "cache", "me@x.org", interval=0, retry_errors=True)
    again._session = _Session([])
    got = again.get("https://example.org/a")
    assert got.status == 200 and "The real page" in got.text and again._session.calls == 0


def test_an_imported_page_is_checked_built_and_served(tmp_path, tasks_file):
    cache = tmp_path / "cache"
    page = tmp_path / "saved.html"
    page.write_text(PAGE.format(title="Saved", text="text saved from the browser"), encoding="utf-8")
    from snapshot.__main__ import main

    assert main(["import-page", "--url", "https://example.org/gone", "--file", str(page), "--cache", str(cache)]) == 0
    fetcher = build.Fetcher(cache, "me@x.org", interval=0, retry_errors=True)
    results = {r["url"]: r["status"] for r in build.check(tasks_file, _OnlyCached(fetcher))}
    assert results["https://example.org/gone"] == "ok"


class _OnlyCached:
    """The fetcher for the pages in the cache, a fake for the rest (the check would otherwise go online)."""

    def __init__(self, fetcher):
        self.fetcher, self.fake = fetcher, FakeFetcher()
        self.skip_hosts = set()

    def get(self, url, params=None, *, api=True):
        path = self.fetcher._key(url, params)
        return self.fetcher.get(url, params, api=api) if path.exists() else self.fake.get(url, params, api=api)


# URLs the tasks only point to loosely


def test_extra_urls_are_read_with_an_optional_task_id(tmp_path):
    path = tmp_path / "extra.txt"
    path.write_text("# a query\nhttps://x.org/a?q=1 SFB-0038  # the planet\n\nhttps://x.org/b\n", encoding="utf-8")
    assert build.load_extra_urls(path) == [("https://x.org/a?q=1", "SFB-0038"), ("https://x.org/b", "extra")]
    assert build.load_extra_urls(None) == []


def test_extra_urls_are_checked_and_built_like_task_sources(tasks_file, tmp_path):
    extra = [("https://example.org/query?q=1&format=csv", "T9")]
    needs = {n.request_key: n for n in build.build_inventory(TASKS, extra)}
    assert needs["https://example.org/query?format=csv&q=1"].task_ids == {"T9"}
    results = {r["url"]: r["status"] for r in build.check(tasks_file, FakeFetcher(), extra=extra)}
    assert results["https://example.org/query?q=1&format=csv"] == "ok"
    build.build(tasks_file, tmp_path / "snap", FakeFetcher(), snapshot_date="2026-11-01", extra=extra)
    assert SnapshotStore(tmp_path / "snap").has("web", "https://example.org/query?format=csv&q=1")


def test_an_address_with_apostrophes_is_kept_whole():
    text = "read it here: https://x.org/TAP/sync?query=select+r+from+ps+where+n='18+Del+b'&format=csv\nFinal answer"
    assert build.urls_in_text(text) == ["https://x.org/TAP/sync?query=select+r+from+ps+where+n='18+Del+b'&format=csv"]
    assert build.urls_in_text("see 'https://x.org/a' and \"https://x.org/b\".") == [
        "https://x.org/a",
        "https://x.org/b",
    ]
