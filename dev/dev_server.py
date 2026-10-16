"""Local viewer and JSON endpoints over ``litsense_core`` (CLAUDE.md s12.8). Manual testing only;
never deployed. Standard library only, so it adds no dependency.

    python dev/dev_server.py --bundle path/to/serve_bundle [--port 8000]

GET  /                 page: pick a demo reader, see their history and recommendations
GET  /health           litsense_core.health()
GET  /demo/<user_id>   a demo reader's ratings (from the bundle) and their recommendations
POST /recommend        body {"state": UserState, "k": 10, "filters": Filters | null}
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import litsense_core  # noqa: E402
from litsense_core.schemas import Filters, RatingEvent, UserState  # noqa: E402
from litsense_core.store import ArtifactStore, get_store  # noqa: E402

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>LitSense dev</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;max-width:960px;background:#fafafa;color:#222}
.row{display:flex;gap:12px;flex-wrap:wrap}.card{background:#fff;border:1px solid #ddd;
border-radius:8px;padding:8px;width:170px}.card img{width:100%;height:220px;object-fit:contain}
.t{font-weight:600;font-size:14px}.a{font-size:12px;color:#555}.r{font-size:12px;margin-top:4px}
h2{font-size:16px;margin-top:24px}select{font-size:15px;padding:4px}
</style></head><body><h1>LitSense dev viewer</h1><p id="h"></p>
<label>Demo reader <select id="u"></select></label>
<h2>Their ratings (newest first)</h2><div id="hist" class="row"></div>
<h2>Recommendations</h2><div id="recs" class="row"></div>
<script>
const card=(b,extra)=>`<div class="card">${b.image?`<img src="${b.image}">`:""}
<div class="t">${b.title}</div><div class="a">${b.author||""}</div>
<div class="r">${extra}</div></div>`;
async function show(u){const d=await (await fetch('/demo/'+encodeURIComponent(u))).json();
document.getElementById('hist').innerHTML=d.history.map(b=>card(b,b.stars+'★')).join('');
document.getElementById('recs').innerHTML=d.recs.items.map(i=>card(d.books[i.book_id],
'#'+i.rank+' · '+i.reason+'<br><i>'+i.sources.join(', ')+'</i>')).join('');}
(async()=>{const h=await (await fetch('/health')).json();
document.getElementById('h').textContent=`bundle ${h.bundle_version} · ${h.n_books} books`;
const users=await (await fetch('/demo')).json();const s=document.getElementById('u');
s.innerHTML=users.map(u=>`<option>${u}</option>`).join('');s.onchange=()=>show(s.value);
show(users[0]);})();
</script></body></html>"""


def demo_state(store: ArtifactStore, user_id: str) -> UserState:
    """A demo reader's bundled ratings as a ``UserState``."""
    row = store.corpus_user(user_id)
    if row is None:
        raise KeyError(user_id)
    h = store.corpus_history
    lo, hi = int(h["offsets"][row]), int(h["offsets"][row + 1])
    ratings = [
        RatingEvent(book_id=int(i), stars=float(s), ts=int(t))
        for i, s, t in zip(h["items"][lo:hi], h["stars"][lo:hi], h["ts"][lo:hi], strict=True)
    ]
    return UserState(corpus_user_id=user_id, ratings=ratings)


def book(store: ArtifactStore, i: int) -> dict[str, Any]:
    t = store.text
    return {"book_id": i, "title": t["title"][i], "author": t["author"][i], "image": t["image"][i]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: Any, status: int = 200, kind: str = "application/json") -> None:
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        store = get_store()
        if self.path == "/":
            return self._send(PAGE, kind="text/html; charset=utf-8")
        if self.path == "/health":
            return self._send(litsense_core.health())
        if self.path == "/demo":
            return self._send(store.corpus_ids[:50])
        if self.path.startswith("/demo/"):
            try:
                state = demo_state(store, self.path.removeprefix("/demo/"))
            except KeyError:
                return self._send({"error": "unknown demo user"}, 404)
            recs = litsense_core.recommend(state, k=10)
            history = [book(store, r.book_id) | {"stars": r.stars} for r in state.ratings[::-1]]
            books = {i.book_id: book(store, i.book_id) for i in recs.items}
            return self._send({"history": history[:20], "recs": recs.model_dump(), "books": books})
        self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if self.path != "/recommend":
            return self._send({"error": "not found"}, 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            state = UserState.model_validate(body.get("state", {}))
            filters = Filters.model_validate(body["filters"]) if body.get("filters") else None
            recs = litsense_core.recommend(state, k=int(body.get("k", 10)), filters=filters)
        except ValueError as e:  # pydantic ValidationError is a ValueError
            return self._send({"error": str(e)}, 400)
        self._send(recs.model_dump())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", required=True, help="unpacked serve_bundle directory")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    litsense_core.load_store(args.bundle)
    print(f"LitSense dev server on http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
