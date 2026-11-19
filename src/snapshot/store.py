"""Read-only access to a snapshot directory."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import schema

_LIGHT_COLUMNS = ("doc_id", "service", "request_key", "url", "lang", "status", "title", "snippet")


class SnapshotError(RuntimeError):
    """The snapshot directory is missing, corrupt or of an unsupported schema."""


class SnapshotStore:
    """Lookup and iteration over the servable items of a snapshot.

    Args:
        data_dir:        Snapshot directory (holding `manifest.json` + `snapshot.parquet`).
        verify_checksum: Recompute the sha256 of `snapshot.parquet` and compare it with
                         the manifest before serving (default True).
    """

    def __init__(self, data_dir: str | Path, verify_checksum: bool = True):
        schema.require_local_extra("pyarrow")
        import pyarrow.parquet as pq

        self.data_dir = Path(data_dir)
        try:
            self.manifest = schema.read_manifest(self.data_dir)
        except FileNotFoundError as exc:
            raise SnapshotError(str(exc)) from exc
        version = self.manifest.get("schema_version")
        if version != schema.SCHEMA_VERSION:
            raise SnapshotError(
                f"Snapshot schema version {version!r} is not supported (expected {schema.SCHEMA_VERSION})."
            )
        path = self.data_dir / schema.SNAPSHOT_FILE
        if not path.exists():
            raise SnapshotError(f"{path} not found.")
        expected = ((self.manifest.get("files") or {}).get(schema.SNAPSHOT_FILE) or {}).get("sha256")
        self.sha256 = expected
        if verify_checksum:
            actual = schema.sha256_file(path)
            if expected and actual != expected:
                raise SnapshotError(
                    f"{path.name} sha256 {actual} does not match the manifest ({expected}); "
                    "the snapshot is corrupt or was modified."
                )
            self.sha256 = actual

        table = pq.read_table(path)
        missing = [c for c in schema.SNAPSHOT_COLUMNS if c not in table.column_names]
        if missing:
            raise SnapshotError(f"{path.name} lacks columns: {', '.join(missing)}")
        self._table = table.combine_chunks()
        services = self._table.column("service").to_pylist()
        keys = self._table.column("request_key").to_pylist()
        self._index: dict[tuple[str, str], int] = {}
        self._by_service: dict[str, list[int]] = {}
        for i, (service, key) in enumerate(zip(services, keys, strict=True)):
            if (service, key) in self._index:
                raise SnapshotError(f"duplicate request_key {service}:{key}")
            self._index[(service, key)] = i
            self._by_service.setdefault(service, []).append(i)
        self._columns = {name: self._table.column(name) for name in self._table.column_names}

    @property
    def snapshot_date(self) -> str | None:
        return self.manifest.get("snapshot_date")

    def summary(self) -> dict[str, Any]:
        out = schema.snapshot_summary(self.manifest)
        out["snapshot_sha256"] = self.sha256
        out["rows"] = len(self)
        return out

    def __len__(self) -> int:
        return self._table.num_rows

    def _row(self, i: int, columns=None) -> dict[str, Any]:
        names = columns or self._table.column_names
        return {name: self._columns[name][i].as_py() for name in names}

    def get(self, service: str, request_key: str) -> dict[str, Any] | None:
        """The row stored under `(service, request_key)`, or None."""
        i = self._index.get((service, request_key))
        return None if i is None else self._row(i)

    def has(self, service: str, request_key: str) -> bool:
        return (service, request_key) in self._index

    def keys(self, service: str) -> list[str]:
        col = self._columns["request_key"]
        return [col[i].as_py() for i in self._by_service.get(service, [])]

    def iter_rows(self, service: str, columns=None) -> Iterator[dict[str, Any]]:
        """Iterate a service's rows in storage order (sorted by doc_id at build time)."""
        for i in self._by_service.get(service, []):
            yield self._row(i, columns)

    def light_rows(self, service: str) -> Iterator[dict[str, Any]]:
        """Rows without the (possibly large) `body`/`markdown` columns."""
        return self.iter_rows(service, _LIGHT_COLUMNS + ("search_text",))
