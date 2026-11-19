"""Running the local server."""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import os
import socket
import threading
import time
from pathlib import Path

from . import schema

logger = logging.getLogger(__name__)


def resolve_data_dir(data: str | Path | None = None) -> Path:
    """Locate a snapshot: the explicit `data` path, else `./local_snapshot/`."""
    if data:
        path = Path(data).expanduser()
        if not (path / schema.MANIFEST_FILE).exists():
            raise FileNotFoundError(f"--data {path}: no {schema.MANIFEST_FILE} found there.")
        return path
    default = Path.cwd() / schema.DEFAULT_DATA_DIR
    if (default / schema.MANIFEST_FILE).exists():
        return default
    raise FileNotFoundError(f"No local snapshot: pass --data, or put one in ./{schema.DEFAULT_DATA_DIR}/.")


def ensure_no_proxy(url: str) -> None:
    """Make sure requests/httpx never send local-server traffic through an HTTP proxy.

    Cluster nodes often export `HTTP(S)_PROXY`; without a matching `NO_PROXY` entry the
    loopback requests would be forwarded to the proxy and fail.
    """
    import urllib.parse

    host = urllib.parse.urlsplit(url).hostname or "127.0.0.1"
    for var in ("NO_PROXY", "no_proxy"):
        current = [h.strip() for h in os.environ.get(var, "").split(",") if h.strip()]
        for entry in (host, "127.0.0.1", "localhost"):
            if entry not in current:
                current.append(entry)
        os.environ[var] = ",".join(current)


class BackgroundServer:
    """A uvicorn server running in a daemon thread (single-node runs, no extra process).

    Use as a context manager or call :meth:`start` / :meth:`stop`.  With `port=0` a
    free port is chosen by the OS; :attr:`url` gives the address to pass to the tools.
    """

    def __init__(
        self,
        store_or_path,
        host: str = schema.DEFAULT_LOCAL_HOST,
        port: int = 0,
        log_level: str = "warning",
    ):
        schema.require_local_extra("fastapi", "uvicorn")
        from .server import create_app

        self.app = create_app(store_or_path)
        self.host = host
        self.port = port
        self.log_level = log_level
        self._server = None
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    @property
    def store(self):
        return self.app.state.store

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, timeout: float = 30.0) -> BackgroundServer:
        import uvicorn

        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        self.port = sock.getsockname()[1]
        self._socket = sock
        config = uvicorn.Config(self.app, log_level=self.log_level, access_log=False, lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [sock]},
            daemon=True,
            name="sciflawbench-local-server",
        )
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("local server thread exited during start-up")
            if time.monotonic() > deadline:
                raise TimeoutError("local server did not start in time")
            time.sleep(0.02)
        ensure_no_proxy(self.url)
        return self

    def stop(self, timeout: float = 10.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.close()
        self._server = self._thread = self._socket = None

    def __enter__(self) -> BackgroundServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def wait_for_server(url: str, timeout: float = 30.0) -> dict:
    """Poll `{url}/healthz` until the server answers; return its manifest summary."""
    import requests

    ensure_no_proxy(url)
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{url.rstrip('/')}/healthz", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"Local server at {url} is not reachable: {last}")


@contextlib.contextmanager
def block_outbound_network(allow_hosts: tuple[str, ...] = ()):
    """Refuse every non-loopback socket connection inside the block.

    Used by `verify` and the tests to prove that local mode makes no live request: any
    attempt raises `ConnectionRefusedError` naming the address.
    """
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def allowed(address) -> bool:
        host = address[0] if isinstance(address, tuple) else address
        if host in allow_hosts or host == "localhost":
            return True
        try:
            return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
        except ValueError:
            return False

    def guarded_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not allowed(address):
            raise ConnectionRefusedError(f"outbound network blocked in local mode: {address!r}")
        return original_connect(self, address)

    def guarded_connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not allowed(address):
            raise ConnectionRefusedError(f"outbound network blocked in local mode: {address!r}")
        return original_connect_ex(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in (None, "localhost") and not allowed((host,)):
            raise socket.gaierror(f"name resolution blocked in local mode: {host!r}")
        return original_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_getaddrinfo
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.getaddrinfo = original_getaddrinfo
