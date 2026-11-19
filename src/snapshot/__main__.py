"""`python -m snapshot serve --data PATH [--host 127.0.0.1] [--port 8765]`: serve a snapshot until interrupted."""

import argparse
import json
import time

from . import schema
from .runtime import BackgroundServer, resolve_data_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m snapshot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="serve a snapshot over HTTP")
    serve.add_argument("--data", help=f"snapshot directory (default ./{schema.DEFAULT_DATA_DIR}/)")
    serve.add_argument("--host", default=schema.DEFAULT_LOCAL_HOST)
    serve.add_argument("--port", type=int, default=schema.DEFAULT_LOCAL_PORT)
    args = parser.parse_args(argv)

    with BackgroundServer(resolve_data_dir(args.data), host=args.host, port=args.port, log_level="info") as server:
        print(f"Serving at {server.url}")
        print(json.dumps(server.store.summary(), indent=2))
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
