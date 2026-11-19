"""Snapshot schema, manifest helpers and shared constants for local mode.

Adapted from the EvidenceQuest local snapshot (Polygl0t/EvidenceQuest), reduced to the services the tools use."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

DEFAULT_LOCAL_HOST = "127.0.0.1"
DEFAULT_LOCAL_PORT = 8765
DEFAULT_LOCAL_URL = f"http://{DEFAULT_LOCAL_HOST}:{DEFAULT_LOCAL_PORT}"

SNAPSHOT_FILE = "snapshot.parquet"
PROVENANCE_FILE = "provenance.parquet"
GOLDEN_FILE = "golden.parquet"
MANIFEST_FILE = "manifest.json"
# What a published snapshot consists of (and what `download` fetches).
SNAPSHOT_FILES = (SNAPSHOT_FILE, PROVENANCE_FILE, GOLDEN_FILE, MANIFEST_FILE)

# Default local data directory (checked after an explicit --data path).
DEFAULT_DATA_DIR = "local_snapshot"

# Services stored in `snapshot.parquet`.
WEB = "web"
WIKIPEDIA = "wikipedia"
ARXIV = "arxiv"
SERVICES = (WEB, WIKIPEDIA, ARXIV)

# Roles recorded in `provenance.parquet` (never in `snapshot.parquet`).
ROLES = ("gold", "prompt", "confounder", "harvested", "bulk")

# Column name -> Arrow type name.  Kept as strings so this module imports without
# pyarrow; `arrow_schema()` turns them into a real schema.
SNAPSHOT_COLUMNS: dict[str, str] = {
    "doc_id": "string",
    "service": "string",
    "request_key": "string",
    "url": "string",
    "lang": "string",
    "status": "int32",
    "content_type": "string",
    "body": "string",
    "markdown": "string",
    "title": "string",
    "snippet": "string",
    "search_text": "string",
    "fetched_at": "timestamp",
}

PROVENANCE_COLUMNS: dict[str, str] = {
    "doc_id": "string",
    "service": "string",
    "request_key": "string",
    "role": "string",
    "task_ids": "list<string>",
    "source_url": "string",
    "note": "string",
    "leak_flag": "bool",
    "leak_context": "string",
}

GOLDEN_COLUMNS: dict[str, str] = {
    "kind": "string",
    "service": "string",
    "url": "string",
    "status": "int32",
    "content_type": "string",
    "body": "string",
    "fetched_at": "timestamp",
}


class LocalExtraMissingError(ImportError):
    """Raised when local mode is used without the optional `local` extra."""


def require_local_extra(*modules: str) -> None:
    """Import-check the local-mode dependencies and fail with an actionable message."""
    missing = []
    for name in modules or ("fastapi", "uvicorn", "pyarrow"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise LocalExtraMissingError(
            "Local mode needs the optional 'local' extra (missing: "
            + ", ".join(missing)
            + "). Install it with `pip install -e '.[local]'` or `uv sync --extra local`."
        )


def arrow_schema(columns: dict[str, str]):
    """Build a pyarrow schema from one of the `*_COLUMNS` mappings."""
    import pyarrow as pa

    types = {
        "string": pa.string(),
        "int32": pa.int32(),
        "bool": pa.bool_(),
        "timestamp": pa.timestamp("us", tz="UTC"),
        "list<string>": pa.list_(pa.string()),
    }
    return pa.schema([(name, types[kind]) for name, kind in columns.items()])


def doc_id(service: str, request_key: str) -> str:
    """Stable id of a servable item: sha256 of `service` + NUL + `request_key` (32 hex)."""
    return hashlib.sha256(f"{service}\0{request_key}".encode()).hexdigest()[:32]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(data_dir: str | Path) -> dict[str, Any]:
    """Load `manifest.json` from a snapshot directory."""
    path = Path(data_dir) / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"No {MANIFEST_FILE} in {data_dir}; is this a snapshot directory?")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_manifest(data_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write `manifest.json` with stable key order (so it diffs cleanly)."""
    path = Path(data_dir) / MANIFEST_FILE
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def snapshot_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest fields recorded in run_config.json / run_manifest.json."""
    files = manifest.get("files") or {}
    return {
        "schema_version": manifest.get("schema_version"),
        "snapshot_date": manifest.get("snapshot_date"),
        "snapshot_sha256": (files.get(SNAPSHOT_FILE) or {}).get("sha256"),
        "rows_per_service": manifest.get("rows_per_service") or {},
    }


def check_files(data_dir: str | Path, manifest: dict[str, Any] | None = None) -> list[str]:
    """Compare every file listed in the manifest against its recorded sha256.

    Returns a list of problems (empty when all checksums match).  Files the manifest
    does not list are ignored; a listed file that is missing is a problem.
    """
    data_dir = Path(data_dir)
    manifest = manifest if manifest is not None else read_manifest(data_dir)
    problems = []
    for name, info in sorted((manifest.get("files") or {}).items()):
        path = data_dir / name
        if not path.exists():
            problems.append(f"{name}: listed in the manifest but missing")
            continue
        actual = sha256_file(path)
        if actual != info.get("sha256"):
            problems.append(f"{name}: sha256 {actual[:12]}… != manifest {str(info.get('sha256'))[:12]}…")
    return problems
