"""FastAPI mock server (read-only).

| Route                            | Mirrors                         | Used by           |
|----------------------------------|---------------------------------|-------------------|
| GET|POST /serper/search          | Serper                          | `web_search`      |
| GET  /serpapi/search.json        | SerpAPI                         |                   |
| GET  /web/fetch?url=             | an arbitrary page               | `visit_webpage`   |
| GET  /wikipedia/{lang}/w/api.php | MediaWiki action API            | `wikipedia_search`|
| GET  /arxiv/api/query            | arXiv export API (Atom)         | `arxiv_search`    |
| GET  /healthz                    | manifest summary                |                   |

Adapted from the EvidenceQuest local snapshot (Polygl0t/EvidenceQuest).
"""

from pathlib import Path

from . import schema
from .services import LocalServices
from .store import SnapshotStore

# Route prefixes, shared with the client-side URL rewriting in `tools.py`.
PREFIX_SERPAPI = "/serpapi/search.json"
PREFIX_SERPER = "/serper/search"
PREFIX_WEB_FETCH = "/web/fetch"
PREFIX_WIKIPEDIA = "/wikipedia"
PREFIX_ARXIV = "/arxiv/api/query"


def create_app(store_or_path, *, warm: bool = True):
    """Build the FastAPI app over a :class:`SnapshotStore` (or a snapshot directory).

    `warm=True` builds every search index up front, so the first request of a run is not
    slower than the others and failures surface at start-up.
    """
    schema.require_local_extra("fastapi")
    from fastapi import FastAPI, Request, Response

    store = store_or_path if isinstance(store_or_path, SnapshotStore) else SnapshotStore(Path(store_or_path))
    services = LocalServices(store)
    if warm:
        _ = (services.web_index, services.wiki, services.arxiv)

    app = FastAPI(title="SciFlawBench local snapshot", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.state.services = services

    def send(result) -> Response:
        status, content_type, body = result
        return Response(content=body.encode("utf-8"), status_code=status, media_type=content_type)

    def params(request: Request) -> dict[str, str]:
        # Last value wins for repeated keys, like most of the emulated services.
        return dict(request.query_params.items())

    @app.get("/healthz")
    def healthz():
        return services.health()

    @app.get(PREFIX_SERPAPI)
    def serpapi(request: Request):
        return send(services.serpapi(params(request)))

    @app.get(PREFIX_SERPER)
    def serper_get(request: Request):
        return send(services.serper(params(request)))

    @app.post(PREFIX_SERPER)
    async def serper_post(request: Request):
        try:
            payload = await request.json()
        except ValueError:
            payload = {}
        merged = {**params(request), **(payload if isinstance(payload, dict) else {})}
        return send(services.serper(merged))

    @app.get(PREFIX_WEB_FETCH)
    def web_fetch(url: str = ""):
        return send(services.fetch(url))

    @app.get(PREFIX_WIKIPEDIA + "/{lang}/w/api.php")
    def wikipedia(lang: str, request: Request):
        return send(services.wikipedia_api(lang, params(request)))

    @app.get(PREFIX_ARXIV)
    def arxiv(request: Request):
        return send(services.arxiv_query(params(request)))

    return app
