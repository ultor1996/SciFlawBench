"""Deterministic local environment: a frozen snapshot of web sources, served by a FastAPI server.

In local mode the network tools (`web_search`, `visit_webpage`, `wikipedia_search`, `arxiv_search`) take a `base_url`
(see `tools.localmode`) and talk to this server instead of the internet. Tool names, inputs, output formats and error
messages stay the same, so a run can be repeated with the same pages.

Adapted from the EvidenceQuest local snapshot (Polygl0t/EvidenceQuest).

Layout of a snapshot directory:

- `snapshot.parquet`   one row per servable item (read by the server);
- `provenance.parquet` which task and source an item came from, and its role (builder and audit only);
- `golden.parquet`     raw live responses kept for shape checks (builder and verify only);
- `manifest.json`      schema version, snapshot date, per-file sha256, row counts, library versions.

The server needs the optional `local` extra: `pip install -e '.[local]'`.
"""

from .schema import (
    DEFAULT_LOCAL_HOST,
    DEFAULT_LOCAL_PORT,
    DEFAULT_LOCAL_URL,
    SCHEMA_VERSION,
    LocalExtraMissingError,
    require_local_extra,
)

__all__ = [
    "DEFAULT_LOCAL_HOST",
    "DEFAULT_LOCAL_PORT",
    "DEFAULT_LOCAL_URL",
    "SCHEMA_VERSION",
    "LocalExtraMissingError",
    "require_local_extra",
]
