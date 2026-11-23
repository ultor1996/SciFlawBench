"""Build a snapshot from a task file: fetch every source the tasks cite once, and write the snapshot directory.

Adapted from the EvidenceQuest builder (Polygl0t/EvidenceQuest, `tools/build_local_snapshot.py`), reduced to what the
SciFlawBench tools use and to the first stage of the build: the pages the tasks themselves point to.

What is stored, and with which role in `provenance.parquet`:

- `gold`:   every URL in a task's `sources`, fetched as the matching tool would receive it: a Wikipedia article (the
            MediaWiki `prop=info` and `prop=extracts` answers, plus the article HTML for `visit_webpage`), an arXiv
            paper (the Atom feed for `id_list=<id>` plus its abstract page), and any other URL as the page itself;
- `prompt`: URLs written in a task's own text, which the agent can see and may visit;
- `gold` (note "validators"): URLs named in a task's validators (answer keys and rubrics), kept so a visit to the
            page the key refers to is not a 404.

Every raw response is cached on disk (`--cache`) before it is used, so an interrupted build resumes where it stopped and
the same cache rebuilds the same snapshot offline (`--offline`). A request that cannot be fetched is written to
`build_review.md` instead of ending the build, and a host that stops answering is given up after a few attempts.

Two steps are meant to be run in turn. `check` asks for every needed source and writes `check_results.json` and
`check_report.md` with the ones that failed (a dead link, a site that is down, a missing page), so they can be fixed
before anything is frozen. Then `build --checked check_results.json` keeps what worked, leaves out what failed, and
lists those in `build_review.md`. A failed source that really is that way live (a page that is gone for good) is
kept as the error the agent would see, by listing its URL in an `--accept` file.

Later stages (not here yet): confounders next to each gold source, an answer-leak scan, and `--harvest` of the pages
agents visited in earlier runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import rows as R
from . import schema
from .canon import canonical_arxiv_id, canonical_url, canonical_wiki_title, classify_url

log = logging.getLogger("snapshot.build")

ARXIV_API = "https://export.arxiv.org/api/query"
WIKI_INFO_PROPS = (
    "protection|talkid|watched|watchers|visitingwatchers|notificationtimestamp|subjectid|"
    "url|readable|preload|displaytitle|varianttitles"
)
# Minimum seconds between two requests to the same host (arXiv asks for 3 s).
HOST_INTERVAL = {"export.arxiv.org": 3.0, "arxiv.org": 3.0}
DEFAULT_INTERVAL = 0.2
# Roles whose pages are expected to exist: a failed fetch of one is a recorded miss.
EXPECTED_ROLES = ("gold", "prompt", "harvested")
CHECK_RESULTS_FILE = "check_results.json"

# An address may hold balanced brackets (`/wiki/Irati_(Paran%C3%A1)`); a lone closing one ends it.
_URL_IN_TEXT = re.compile(r"https?://(?:[^\s()\[\]\"'<>]|\([^\s()\[\]\"'<>]*\))+")
_TRAILING_PUNCTUATION = ".,;:!?'\"*"


@dataclass
class Fetched:
    url: str
    status: int
    content_type: str
    text: str
    fetched_at: str

    def json(self):
        return json.loads(self.text)


class FetchError(RuntimeError):
    """A request could not be answered (network failure, host down, not cached offline). Recorded as a miss."""


class OfflineCacheMiss(FetchError):
    pass


class HostUnavailable(FetchError):
    """The host is skipped (`--skip-host`) or stopped answering (circuit breaker open)."""


class Fetcher:
    """GET with an on-disk cache, per-host spacing, retries and a contact User-Agent.

    Args:
        cache_dir:       Where raw responses are stored (one JSON file per request).
        contact:         E-mail put in the User-Agent (arXiv and Wikipedia ask for one).
        offline:         Only serve from the cache; a miss raises :class:`OfflineCacheMiss`.
        page_user_agent: User-Agent for plain page fetches. `None` sends requests' default, exactly like the live
                         `visit_webpage` tool.
        skip_hosts:      Hosts never contacted: every request to them raises :class:`HostUnavailable` at once.
        down_after:      Consecutive network failures (no HTTP answer at all) after which a host is declared down for
                         the rest of the build. A host that answers (any status) resets the count.
        interval:        Seconds between two requests to one host, for every host (default: 3 s for arXiv, 0.2 s else).
        retry_errors:    Ask again for a cached answer that was an error (status other than 200), for instance a site
                         that was down during the first build. Cached good answers are still used.
        connect_timeout: Seconds to wait for the connection (a silent host costs this much per attempt).
    """

    def __init__(
        self,
        cache_dir: Path,
        contact: str,
        offline: bool = False,
        page_user_agent: str | None = None,
        retries: int = 3,
        skip_hosts=(),
        down_after: int = 3,
        interval: float | None = None,
        connect_timeout: float = 15.0,
        retry_errors: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.retries = retries
        self.api_user_agent = f"SciFlawBench-snapshot-builder/1.0 (mailto:{contact})"
        self.page_user_agent = page_user_agent
        self.interval = interval
        self.retry_errors = retry_errors
        self._last: dict[str, float] = {}
        self._session = None
        self.stats: dict[str, int] = defaultdict(int)
        self.skip_hosts = {h.lower() for h in skip_hosts}
        self.down_after = down_after
        self.connect_timeout = connect_timeout
        self._failures: dict[str, int] = defaultdict(int)
        self.down_hosts: set[str] = set()

    @staticmethod
    def host(url: str) -> str:
        return urllib.parse.urlsplit(url).netloc.lower()

    def _key(self, url: str, params: dict | None) -> Path:
        ident = json.dumps(
            ["GET", url, sorted((params or {}).items())], sort_keys=True, ensure_ascii=False, default=str
        )
        digest = hashlib.sha256(ident.encode()).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def get(self, url: str, params: dict | None = None, *, api: bool = True) -> Fetched:
        host = self.host(url)
        if host in self.skip_hosts:
            raise HostUnavailable(f"{host} is skipped (not contacted): {url}")
        path = self._key(url, params)
        if path.exists():
            cached = Fetched(**json.loads(path.read_text(encoding="utf-8")))
            if cached.status == 200 or self.offline or not self.retry_errors:
                self.stats["cache_hits"] += 1
                return cached
            self.stats["retried_errors"] += 1
        if self.offline:
            raise OfflineCacheMiss(f"GET {url} {params or ''} is not in the cache")
        if host in self.down_hosts:
            raise HostUnavailable(f"{host} is down (no answer to earlier requests): {url}")
        result = self._fetch(url, params, api=api)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result.__dict__, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return result

    def put(self, url: str, text: str, *, status: int = 200, content_type: str = "text/html") -> Path:
        """Store a page obtained some other way (saved from a browser) as the answer to `GET url`.

        For sites that refuse scripts. The build and the check then find it in the cache like any fetched page.
        """
        path = self._key(url, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        got = Fetched(url, status, content_type, text, datetime.now(UTC).isoformat())
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(got.__dict__, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def _fetch(self, url: str, params: dict | None, *, api: bool) -> Fetched:
        import requests

        if self._session is None:
            self._session = requests.Session()
        host = self.host(url)
        headers = {}
        agent = self.api_user_agent if api else self.page_user_agent
        if agent:
            headers["User-Agent"] = agent
        spacing = self.interval if self.interval is not None else HOST_INTERVAL.get(host, DEFAULT_INTERVAL)
        last_exc = None
        for attempt in range(self.retries + 1):
            wait = spacing - (time.monotonic() - self._last.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            try:
                response = self._session.get(url, params=params, headers=headers, timeout=(self.connect_timeout, 60))
                self._last[host] = time.monotonic()
            except requests.RequestException as exc:
                self._last[host] = time.monotonic()
                last_exc = exc
                self._failures[host] += 1
                if self._failures[host] >= self.down_after:
                    self.down_hosts.add(host)
                    log.warning(
                        "%s: %d network failures in a row; treating the host as down (%s).",
                        host,
                        self._failures[host],
                        exc,
                    )
                    raise HostUnavailable(f"{host} is down: GET {url}: {exc}") from exc
                time.sleep(0 if self.interval == 0 else 2**attempt)
                continue
            self._failures[host] = 0
            if response.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                time.sleep(0 if self.interval == 0 else float(response.headers.get("Retry-After") or 2**attempt * 2))
                continue
            self.stats["fetched"] += 1
            return Fetched(
                url=response.url,
                status=response.status_code,
                content_type=response.headers.get("Content-Type", ""),
                text=response.text,
                fetched_at=datetime.now(UTC).isoformat(),
            )
        raise FetchError(f"GET {url} failed after {self.retries + 1} attempts: {last_exc}")


class Collector:
    """What the build has gathered: rows, provenance, and the requests that could not be answered."""

    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = {}
        self.provenance: dict[tuple[str, str, str], dict] = {}
        self.misses: list[dict] = []
        self.skipped: list[Need] = []  # left out on purpose: they failed the check

    def add(self, row: dict, role: str, task_ids=(), source_url: str = "", note: str = "") -> dict:
        self.rows.setdefault((row["service"], row["request_key"]), row)
        pkey = (row["doc_id"], role, source_url)
        if pkey in self.provenance:
            prev = self.provenance[pkey]
            prev["task_ids"] = sorted(set(prev["task_ids"]) | set(task_ids))
        else:
            self.provenance[pkey] = R.provenance_row(row, role, task_ids, source_url, note)
        return row

    def has(self, service: str, key: str) -> bool:
        return (service, key) in self.rows

    def miss(self, service: str, key: str, url: str, reason: str, task_ids=()) -> None:
        self.misses.append(
            {"service": service, "request_key": key, "url": url, "reason": reason, "task_ids": sorted(set(task_ids))}
        )


def _single_page(data: dict) -> dict | None:
    for page in (((data or {}).get("query") or {}).get("pages") or {}).values():
        return page
    return None


class Builder:
    """Fetches one kind of source at a time and adds it to the collector."""

    def __init__(self, fetcher: Fetcher, collector: Collector):
        self.f = fetcher
        self.c = collector

    def page(self, url: str, role: str, task_ids=(), source_url: str = "", note: str = "") -> dict | None:
        key = canonical_url(url)
        if self.c.has(schema.WEB, key):
            row = self.c.rows[(schema.WEB, key)]
            self.c.add(row, role, task_ids, source_url, note)
            return row
        try:
            got = self.f.get(url, api=False)
        except OfflineCacheMiss:
            raise
        except FetchError as exc:
            if role in EXPECTED_ROLES:
                self.c.miss(schema.WEB, key, url, str(exc), task_ids)
            return None
        # A failed fetch (404, 403, ...) is stored too: the agent would have seen that error live, and the
        # server answers it the same way.
        row = R.web_row(url, got.status, got.content_type, got.text, fetched_at=got.fetched_at)
        self.c.add(row, role, task_ids, source_url, note)
        if got.status != 200 and role in EXPECTED_ROLES:
            self.c.miss(schema.WEB, key, url, f"HTTP {got.status}", task_ids)
        # Requests followed a redirect: the final URL serves the same page.
        if canonical_url(got.url) != key and not self.c.has(schema.WEB, canonical_url(got.url)):
            alias = R.web_row(got.url, got.status, got.content_type, got.text, fetched_at=got.fetched_at)
            self.c.add(alias, role, task_ids, source_url, note)
        return row

    def _wapi(self, lang: str, params: dict) -> tuple[dict, str]:
        got = self.f.get(f"https://{lang}.wikipedia.org/w/api.php", {**params, "format": "json"})
        return (got.json() if got.status == 200 else {}), got.fetched_at

    def wiki_page(self, lang: str, title: str, role: str, task_ids=(), source_url: str = "", note: str = ""):
        """Store one article (info, extracts, redirects, and its HTML page); return its title."""
        norm = canonical_wiki_title(title)
        info_resp, stamp = self._wapi(
            lang, {"action": "query", "prop": "info", "titles": norm, "inprop": WIKI_INFO_PROPS, "redirects": 1}
        )
        info = _single_page(info_resp)
        if not info or "missing" in info or "invalid" in info:
            if role in EXPECTED_ROLES:
                self.c.miss(schema.WIKIPEDIA, f"{lang}:{norm}", source_url or norm, "missing page", task_ids)
            return None
        target = info["title"]
        ext_resp, _ = self._wapi(
            lang,
            {
                "action": "query",
                "prop": "extracts",
                "titles": target,
                "explaintext": 1,
                "exsectionformat": "wiki",
                "redirects": 1,
            },
        )
        extracts = _single_page(ext_resp) or {"pageid": info.get("pageid"), "ns": 0, "title": target, "extract": ""}
        red_resp, _ = self._wapi(
            lang, {"action": "query", "prop": "redirects", "titles": target, "rdlimit": "max", "rdnamespace": 0}
        )
        redirects = [r["title"] for r in (_single_page(red_resp) or {}).get("redirects") or []]
        if canonical_wiki_title(norm) != canonical_wiki_title(target):
            redirects.append(norm)
        row = R.wikipedia_row(lang, info, extracts, redirects, fetched_at=stamp)
        self.c.add(row, role, task_ids, source_url, note)
        self.page(
            info.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{target}", role, task_ids, source_url, note
        )
        return target

    def arxiv_paper(self, arxiv_id: str, role: str, task_ids=(), source_url: str = "", note: str = "") -> dict | None:
        key = canonical_arxiv_id(arxiv_id)
        if self.c.has(schema.ARXIV, key):
            row = self.c.rows[(schema.ARXIV, key)]
            self.c.add(row, role, task_ids, source_url, note)
            return row
        got = self.f.get(ARXIV_API, {"id_list": key, "max_results": 1})
        row = R.arxiv_row(key, got.text, fetched_at=got.fetched_at) if got.status == 200 else None
        if row is None or not row["title"]:
            if role in EXPECTED_ROLES:
                self.c.miss(schema.ARXIV, key, source_url or key, f"HTTP {got.status}", task_ids)
            return None
        self.c.add(row, role, task_ids, source_url, note)
        self.page(f"https://arxiv.org/abs/{key}", role, task_ids, source_url, note)
        return row


# ---- inventory: what the tasks need ------------------------------------------------------------------------------


@dataclass
class Need:
    service: str
    request_key: str
    url: str
    role: str
    task_ids: set[str] = field(default_factory=set)
    note: str = ""


def load_tasks(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def urls_in_text(text: str) -> list[str]:
    """The URLs written in a text, without the punctuation that ends a sentence."""
    out = []
    for raw in _URL_IN_TEXT.findall(text or ""):
        out.append(raw.rstrip(_TRAILING_PUNCTUATION))
    return out


def build_inventory(tasks: list[dict]) -> list[Need]:
    """Every distinct `(service, request_key, role)` the tasks cite, with the tasks that cite it, in a fixed order."""
    needs: dict[tuple[str, str, str], Need] = {}

    def add(url: str, role: str, task_id: str, note: str = "") -> None:
        service, key = classify_url(url)
        need = needs.setdefault((service, key, role), Need(service, key, url, role, note=note))
        need.task_ids.add(task_id)

    for task in tasks:
        task_id = str(task.get("task_id") or task.get("id"))
        for source in task.get("sources") or []:
            if source.get("url"):
                add(source["url"], "gold", task_id)
        for url in urls_in_text(task.get("task", "")):
            add(url, "prompt", task_id)
        for url in urls_in_text(json.dumps(task.get("validators") or [], ensure_ascii=False)):
            add(url, "gold", task_id, note="validators")
    return sorted(needs.values(), key=lambda n: (n.role != "gold", n.service, n.request_key))


def fetch_need(builder: Builder, collector: Collector, need: Need) -> None:
    """Fetch one needed source into the collector. A failure becomes a recorded miss, never an exception."""
    ids, args = sorted(need.task_ids), (need.role, sorted(need.task_ids), need.url, need.note)
    try:
        if need.service == schema.WIKIPEDIA:
            lang, title = need.request_key.split(":", 1)
            builder.wiki_page(lang, title, *args)
        elif need.service == schema.ARXIV:
            builder.arxiv_paper(need.request_key, *args)
        else:
            builder.page(need.url, *args)
    except OfflineCacheMiss:
        raise
    except FetchError as exc:  # one unreachable source must not end the build
        log.warning("  miss: %s %s (%s)", need.service, need.request_key, exc)
        collector.miss(need.service, need.request_key, need.url, str(exc), ids)
    except Exception as exc:  # noqa: BLE001 - one odd page must not end a long build either
        log.warning("  miss: %s %s (unexpected %s: %s)", need.service, need.request_key, type(exc).__name__, exc)
        collector.miss(need.service, need.request_key, need.url, f"{type(exc).__name__}: {exc}", ids)


# ---- check: do the sources work? -------------------------------------------------------------------------------------


def check(tasks_path: str | Path, fetcher: Fetcher, *, limit_tasks: int = 0) -> list[dict]:
    """Ask for every needed source, store nothing, and say for each whether it worked.

    The answers go into the fetcher's cache, so the build that follows does not ask again for the ones that worked.
    """
    tasks = load_tasks(tasks_path)
    if limit_tasks:
        tasks = tasks[:limit_tasks]
    inventory = build_inventory(tasks)
    results = []
    for n, need in enumerate(inventory, 1):
        if n % 10 == 0:
            log.info("  checked %d/%d", n, len(inventory))
        collector = Collector()
        fetch_need(Builder(fetcher, collector), collector, need)
        reasons = sorted({str(m["reason"]) for m in collector.misses})
        results.append(
            {
                "service": need.service,
                "request_key": need.request_key,
                "url": need.url,
                "role": need.role,
                "task_ids": sorted(need.task_ids),
                "note": need.note,
                "status": "failed" if reasons else "ok",
                "reason": "; ".join(reasons),
            }
        )
    return results


def _failed_row(result: dict) -> str:
    reason = result["reason"].replace("|", "/").replace("\n", " ")[:160]
    return f"| {result['url']} | {reason} | {', '.join(result['task_ids'])} |"


def write_check(out_dir: str | Path, results: list[dict], tasks_path: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "checked_at": datetime.now(UTC).isoformat(),
        "tasks_file": Path(tasks_path).name,
        "tasks_sha256": hashlib.sha256(Path(tasks_path).read_bytes()).hexdigest(),
        "results": results,
    }
    (out / CHECK_RESULTS_FILE).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = [r for r in results if r["status"] == "failed"]
    lines = [
        "# Source check",
        "",
        f"{len(results)} sources asked for, {len(results) - len(failed)} worked, {len(failed)} failed.",
        "",
    ]
    if failed:
        lines += [
            "## Failed",
            "",
            "| Source | Reason | Tasks |",
            "|---|---|---|",
            *(_failed_row(r) for r in failed),
            "",
            "## What to do with each",
            "",
            "- The address is wrong or moved: fix it in the task file, then run `check` again.",
            "- The site is down for now: run `check` again later (failed sources are asked for again).",
            "- The page is gone for good, and the agent should see that error: write its address in a text file and",
            "  pass it to `build --accept FILE`. The error is then stored as it is.",
            "- Anything else is left out of the snapshot by `build --checked` and is listed in `build_review.md`.",
        ]
    else:
        lines.append("Every source worked.")
    (out / "check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def failed_to_skip(check_file: str | Path, accept_file: str | Path | None = None) -> set[tuple[str, str, str]]:
    """The `(service, request_key, role)` of sources that failed the check and were not accepted."""
    results = json.loads(Path(check_file).read_text(encoding="utf-8"))["results"]
    accepted: set[tuple[str, str]] = set()
    if accept_file:
        for line in Path(accept_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                accepted.add(classify_url(line))
    return {
        (r["service"], r["request_key"], r["role"])
        for r in results
        if r["status"] == "failed" and (r["service"], r["request_key"]) not in accepted
    }


# ---- the build ------------------------------------------------------------------------------------------------------


def build(
    tasks_path: str | Path,
    out_dir: str | Path,
    fetcher: Fetcher,
    *,
    limit_tasks: int = 0,
    snapshot_date: str | None = None,
    skip: set[tuple[str, str, str]] = frozenset(),
) -> dict[str, Any]:
    """Fetch every source the tasks cite and write the snapshot directory; return the manifest."""
    tasks = load_tasks(tasks_path)
    if limit_tasks:
        tasks = tasks[:limit_tasks]
    inventory = build_inventory(tasks)
    collector = Collector()
    builder = Builder(fetcher, collector)
    log.info("Inventory: %d distinct requests from %d tasks.", len(inventory), len(tasks))

    for n, need in enumerate(inventory, 1):
        if n % 10 == 0:
            log.info("  %d/%d", n, len(inventory))
        if (need.service, need.request_key, need.role) in skip:
            collector.skipped.append(need)
            continue
        fetch_need(builder, collector, need)

    extra = {
        "tasks_file": Path(tasks_path).name,
        "tasks_sha256": hashlib.sha256(Path(tasks_path).read_bytes()).hexdigest(),
        "tasks": len(tasks),
        "requests_needed": len(inventory),
        "requests_missed": len(collector.misses),
        "requests_skipped": len(collector.skipped),
        "roles": dict(sorted(_count(p["role"] for p in collector.provenance.values()).items())),
    }
    manifest = R.write_snapshot(
        out_dir,
        list(collector.rows.values()),
        sorted(collector.provenance.values(), key=lambda p: (p["doc_id"], p["role"], p["source_url"])),
        [],
        snapshot_date=snapshot_date,
        extra=extra,
    )
    write_review(Path(out_dir) / "build_review.md", collector, inventory, manifest)
    return manifest


def _count(items) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for item in items:
        out[item] += 1
    return out


def write_review(path: Path, collector: Collector, inventory: list[Need], manifest: dict) -> None:
    lines = [
        "# Build review",
        "",
        f"Snapshot date: {manifest['snapshot_date']}. Rows per service: {manifest['rows_per_service']}.",
        f"Requests needed: {len(inventory)}; not answered: {len(collector.misses)}.",
        "",
    ]
    if collector.misses:
        lines += [
            "## Sources that could not be stored",
            "",
            "| Service | Request | Reason | Tasks |",
            "|---|---|---|---|",
        ]
        for miss in sorted(collector.misses, key=lambda m: (m["service"], m["request_key"])):
            reason = str(miss["reason"]).replace("|", "/").replace("\n", " ")[:160]
            lines.append(f"| {miss['service']} | {miss['url']} | {reason} | {', '.join(miss['task_ids'])} |")
    else:
        lines.append("Every source was stored.")
    if collector.skipped:
        lines += [
            "",
            "## Left out: they failed the check",
            "",
            "Not in the snapshot, so an agent visiting them gets a 404 in local mode. Fix them and run the check",
            "again, or accept the error as it is with `--accept`.",
            "",
            *(f"- {n.url} (tasks {', '.join(sorted(n.task_ids))})" for n in collector.skipped),
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---- command line --------------------------------------------------------------------------------------------------


_PAGE_UA_HELP = (
    "User-Agent for plain pages (default: requests', like the live visit_webpage tool). A browser string gets pages "
    "from sites that refuse scripts, but then the snapshot holds pages the live tool cannot read"
)


def add_arguments(sub) -> None:
    imp = sub.add_parser("import-page", help="store a page saved from your browser as the answer for a URL")
    imp.add_argument("--url", required=True, help="the address the tasks use for the page")
    imp.add_argument("--file", required=True, help="the saved page (HTML or text)")
    imp.add_argument("--cache", default=".cache/local_build", help="where raw responses are kept")
    imp.add_argument("--contact", default="unused@example.org", help=argparse.SUPPRESS)
    inv = sub.add_parser("inventory", help="list the requests a snapshot must hold, without fetching anything")
    inv.add_argument("--tasks", required=True, help="the task file (JSON lines)")
    chk = sub.add_parser("check", help="ask for every needed source and report which ones fail; stores no snapshot")
    chk.add_argument("--tasks", required=True, help="the task file (JSON lines)")
    chk.add_argument(
        "--out", default=schema.DEFAULT_DATA_DIR, help="directory for check_results.json and check_report.md"
    )
    chk.add_argument("--contact", required=True, help="your e-mail, put in the User-Agent (arXiv and Wikipedia ask)")
    chk.add_argument("--cache", default=".cache/local_build", help="where raw responses are kept")
    chk.add_argument("--skip-host", action="append", default=[], help="a host not to contact (repeatable)")
    chk.add_argument("--page-user-agent", help=_PAGE_UA_HELP)
    chk.add_argument("--limit-tasks", type=int, default=0, help="use only the first N tasks (a trial run)")
    bld = sub.add_parser("build", help="fetch the sources the tasks cite and write a snapshot directory")
    bld.add_argument("--tasks", required=True, help="the task file (JSON lines)")
    bld.add_argument("--out", default=schema.DEFAULT_DATA_DIR, help="snapshot directory to write")
    bld.add_argument("--contact", required=True, help="your e-mail, put in the User-Agent (arXiv and Wikipedia ask)")
    bld.add_argument("--cache", default=".cache/local_build", help="where raw responses are kept")
    bld.add_argument("--offline", action="store_true", help="use the cache only; touch no network")
    bld.add_argument(
        "--retry-errors", action="store_true", help="ask again for cached answers that were errors (404, 503, ...)"
    )
    bld.add_argument("--skip-host", action="append", default=[], help="a host not to contact (repeatable)")
    bld.add_argument("--page-user-agent", help=_PAGE_UA_HELP)
    bld.add_argument("--limit-tasks", type=int, default=0, help="use only the first N tasks (a trial run)")
    bld.add_argument("--checked", help="check_results.json from `check`: sources that failed are left out")
    bld.add_argument("--accept", help="text file of URLs (one per line) to keep even though they failed the check")
    bld.add_argument("--snapshot-date", help="date to record (default: today); give it to rebuild identically")


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "inventory":
        needs = build_inventory(load_tasks(args.tasks))
        for service in schema.SERVICES:
            part = [n for n in needs if n.service == service]
            print(f"{service}: {len(part)} requests")
        print(json.dumps([{**n.__dict__, "task_ids": sorted(n.task_ids)} for n in needs], indent=2, ensure_ascii=False))
        return 0
    if args.command == "import-page":
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
        path = Fetcher(Path(args.cache), args.contact).put(args.url, text)
        print(f"Stored {len(text)} characters as the page for {args.url}\n  ({path})")
        print("Run `check` again: this source now counts as working.")
        return 0
    if args.command == "check":
        fetcher = Fetcher(
            Path(args.cache),
            args.contact,
            page_user_agent=args.page_user_agent,
            skip_hosts=args.skip_host,
            retry_errors=True,
        )
        results = check(args.tasks, fetcher, limit_tasks=args.limit_tasks)
        write_check(args.out, results, args.tasks)
        failed = [r for r in results if r["status"] == "failed"]
        print(f"{len(results)} sources checked, {len(failed)} failed.")
        for r in failed:
            print(f"  FAILED {r['url']}  ({r['reason'][:100]})  tasks: {', '.join(r['task_ids'])}")
        print(f"Report: {Path(args.out) / 'check_report.md'}")
        return 0
    if args.accept and not args.checked:
        raise SystemExit("--accept only makes sense together with --checked")
    skip = failed_to_skip(args.checked, args.accept) if args.checked else set()
    fetcher = Fetcher(
        Path(args.cache),
        args.contact,
        offline=args.offline,
        page_user_agent=args.page_user_agent,
        skip_hosts=args.skip_host,
        retry_errors=args.retry_errors,
    )
    manifest = build(
        args.tasks, args.out, fetcher, limit_tasks=args.limit_tasks, snapshot_date=args.snapshot_date, skip=skip
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("snapshot_date", "rows_per_service", "requests_needed", "requests_missed", "requests_skipped")
            }
        )
    )
    print(f"Written to {args.out}; see {Path(args.out) / 'build_review.md'} for what could not be stored.")
    return 0
