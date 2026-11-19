"""A tiny snapshot of three services, written to a temp directory, shared by the snapshot tests."""

import importlib.util

import pytest

# The snapshot server needs the optional `local` extra. Without it these tests are not collected, so the rest of the
# suite runs as usual (a module-level `pytest.importorskip` in a conftest would abort the whole run instead).
_MISSING = [name for name in ("fastapi", "pyarrow", "uvicorn") if importlib.util.find_spec(name) is None]
collect_ignore_glob = ["test_*.py"] if _MISSING else []

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/{id}v1</id>
    <published>{published}</published>
    <updated>{published}</updated>
    <title>{title}</title>
    <summary>{summary}</summary>
    <author><name>{author}</name></author>
    <category term="astro-ph.CO"/>
  </entry>
</feed>"""

PAGE = "<html><head><title>{title}</title></head><body><h1>{title}</h1><p>{text}</p></body></html>"


def _paper(arxiv_id, published, title, summary, author="A. Author"):
    from snapshot import rows

    feed = ATOM.format(id=arxiv_id, published=published, title=title, summary=summary, author=author)
    return rows.arxiv_row(arxiv_id, feed, fetched_at="2026-11-01T00:00:00Z")


def build_rows():
    from snapshot import rows

    stamp = "2026-11-01T00:00:00Z"
    return [
        rows.web_row(
            "https://example.org/desi",
            200,
            "text/html",
            PAGE.format(title="DESI dark energy results", text="DESI reports evolving dark energy at 4.2 sigma."),
            fetched_at=stamp,
        ),
        rows.web_row(
            "https://example.org/gw",
            200,
            "text/html",
            PAGE.format(title="GW190521", text="A heavy binary black hole merger seen by LIGO."),
            fetched_at=stamp,
        ),
        rows.web_row("https://example.org/gone", 404, "text/html", "<p>nope</p>", fetched_at=stamp),
        rows.wikipedia_row(
            "en",
            {"pageid": 1, "ns": 0, "title": "Dark energy", "length": 40},
            {"title": "Dark energy", "extract": "Dark energy is an unknown form of energy driving cosmic expansion."},
            redirects=["Quintessence (physics)"],
            fetched_at=stamp,
        ),
        *[
            rows.wikipedia_row(
                "en",
                {"pageid": 10 + i, "ns": 0, "title": title, "length": 30},
                {"title": title, "extract": text},
                fetched_at=stamp,
            )
            for i, (title, text) in enumerate(
                [
                    ("Black hole", "A black hole is a region of spacetime where gravity is extreme."),
                    ("Neutron star", "A neutron star is the collapsed core of a massive star."),
                    ("Supernova", "A supernova is the explosion of a star."),
                    ("Galaxy", "A galaxy is a gravitationally bound system of stars."),
                ]
            )
        ],
        _paper(
            "2503.14738", "2025-03-18T10:00:00Z", "DESI DR2 results", "Baryon acoustic oscillations and dark energy."
        ),
        _paper("1803.07687", "2018-03-20T10:00:00Z", "Old lensing paper", "Gravitational lensing of quasars."),
    ]


@pytest.fixture(scope="session")
def snapshot_dir(tmp_path_factory):
    from snapshot import rows

    path = tmp_path_factory.mktemp("snapshot")
    rows.write_snapshot(path, build_rows(), snapshot_date="2026-11-01")
    return path


@pytest.fixture(scope="session")
def server(snapshot_dir):
    from snapshot.runtime import BackgroundServer

    with BackgroundServer(snapshot_dir) as running:
        yield running
