import json

import pytest

from snapshot import canon, schema
from snapshot.bm25 import BM25Index
from snapshot.services import LocalServices
from snapshot.store import SnapshotError, SnapshotStore


@pytest.fixture
def services(snapshot_dir):
    return LocalServices(SnapshotStore(snapshot_dir))


# canonical keys


def test_urls_that_differ_only_in_noise_have_one_key():
    a = canon.canonical_url("http://www.Example.org/a/?utm_source=x&b=2&a=1#frag")
    b = canon.canonical_url("https://example.org/a?a=1&b=2")
    assert a == b == "https://example.org/a?a=1&b=2"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://en.wikipedia.org/wiki/Dark_energy", ("wikipedia", "en:Dark energy")),
        ("https://en.m.wikipedia.org/wiki/dark_energy", ("wikipedia", "en:Dark energy")),
        ("https://arxiv.org/abs/1803.07687v2", ("arxiv", "1803.07687")),
        ("https://arxiv.org/pdf/1803.07687.pdf", ("arxiv", "1803.07687")),
    ],
)
def test_classify_url(url, expected):
    assert canon.classify_url(url) == expected


# store


def test_store_reads_the_snapshot_and_checks_the_checksum(snapshot_dir):
    store = SnapshotStore(snapshot_dir)
    assert len(store) == 10 and store.snapshot_date == "2026-11-01"
    assert store.summary()["rows_per_service"] == {"arxiv": 2, "web": 3, "wikipedia": 5}
    assert store.has("web", "https://example.org/desi")


def test_a_modified_snapshot_is_refused(snapshot_dir, tmp_path):
    for item in snapshot_dir.iterdir():
        (tmp_path / item.name).write_bytes(item.read_bytes())
    (tmp_path / schema.SNAPSHOT_FILE).write_bytes((tmp_path / schema.SNAPSHOT_FILE).read_bytes() + b"x")
    with pytest.raises(SnapshotError, match="sha256"):
        SnapshotStore(tmp_path)


def test_the_same_inputs_give_the_same_bytes(snapshot_dir, tmp_path):
    from snapshot import rows

    from .conftest import build_rows

    rows.write_snapshot(tmp_path, build_rows(), snapshot_date="2026-11-01")
    assert schema.sha256_file(tmp_path / schema.SNAPSHOT_FILE) == schema.sha256_file(
        snapshot_dir / schema.SNAPSHOT_FILE
    )


# ranking


def test_bm25_is_deterministic_and_ties_break_by_id():
    index = BM25Index()
    index.add("b", "same title", "same text")
    index.add("a", "same title", "same text")
    index.finalize()
    assert index.search("same") == index.search("same")
    assert [doc for doc, _ in index.search("same")] == ["a", "b"]


# services


def test_search_finds_the_right_page_and_misses_cleanly(services):
    status, _, body = services.serper({"q": "evolving dark energy sigma", "num": "5"})
    assert status == 200
    organic = json.loads(body)["organic"]
    assert organic[0]["link"] == "https://example.org/desi"
    assert json.loads(services.serper({"q": "zebra giraffe"})[2])["organic"] == []


def test_error_pages_are_not_searchable(services):
    links = [r["link"] for r in services.web_results("nope")]
    assert "https://example.org/gone" not in links


def test_fetch_returns_markdown_404_and_wikipedia_pages(services):
    status, _, markdown = services.fetch("https://www.example.org/gw?utm_source=x")
    assert status == 200 and "heavy binary black hole merger" in markdown
    assert services.fetch("https://example.org/missing")[0] == 404
    assert services.fetch("https://example.org/gone")[0] == 404
    assert json.loads(services.fetch("https://example.org/missing")[2]) == {"status": 404, "reason": "Not Found"}


def test_wikipedia_titles_redirects_and_search(services):
    _, _, body = services.wikipedia_api(
        "en", {"action": "query", "titles": "quintessence (physics)", "prop": "extracts|info", "redirects": "1"}
    )
    query = json.loads(body)["query"]
    assert query["redirects"] == [{"from": "Quintessence (physics)", "to": "Dark energy"}]
    assert next(iter(query["pages"].values()))["extract"].startswith("Dark energy is")
    hits = json.loads(
        services.wikipedia_api("en", {"action": "query", "list": "search", "srsearch": "cosmic expansion"})[2]
    )
    assert hits["query"]["search"][0]["title"] == "Dark energy"
    assert (
        json.loads(services.wikipedia_api("en", {"action": "query", "titles": "Nothing here"})[2])["query"]["pages"][
            "-1"
        ]["missing"]
        == ""
    )


def _ids(services, **params):
    body = services.arxiv_query(params)[2]
    return [part.split("v1")[0].rsplit("/", 1)[-1] for part in body.split("<id>http://arxiv.org/abs/")[1:]]


def test_arxiv_id_lookup_and_search(services):
    assert _ids(services, id_list="1803.07687") == ["1803.07687"]
    assert _ids(services, search_query="(all:dark AND all:energy)", sortBy="relevance") == ["2503.14738"]


def test_arxiv_submitted_date_filter_and_sorting(services):
    recent = "(all:desi) AND submittedDate:[202501010000 TO 299912312359]"
    assert _ids(services, search_query=recent) == ["2503.14738"]
    old = "(all:lensing) AND submittedDate:[202501010000 TO 299912312359]"
    assert _ids(services, search_query=old) == []
    both = "(all:desi OR all:lensing)"
    assert _ids(services, search_query=both, sortBy="submittedDate", sortOrder="ascending") == [
        "1803.07687",
        "2503.14738",
    ]
    assert _ids(services, search_query=both, sortBy="submittedDate", sortOrder="descending")[0] == "2503.14738"
