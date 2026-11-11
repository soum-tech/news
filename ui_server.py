"""
The news UI server: serves news/ui/index.html and a small JSON API over the stored data (no extra packages).

    python news/ui_server.py                 then open http://localhost:8001      (NEWS_PORT changes the port)

API (GET):
    /api/classes                              classes that have stored days
    /api/days?class=sports                    stored days with article counts
    /api/topics?class=sports&date=2026-11-11  the day's articles grouped by sport: [{"name", "count", "articles": [...]}]
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import store  # noqa: E402

UI_DIR = config.NEWS / "ui"
PORT = int(os.getenv("NEWS_PORT", "8001"))
# fields of an article the page needs (the full text is left out to keep the page light)
KEEP = ("id", "paper", "edition", "paper_date", "page", "headline", "dateline_place", "dateline_date",
        "facts", "meaning", "people", "suggestions", "checks", "text")


def classes() -> list[str]:
    return sorted(p.name for p in config.STORE.iterdir() if p.is_dir()) if config.STORE.exists() else []


def days(cls: str) -> list[dict]:
    out = []
    for d in store.days(cls):
        arts = store.load_day(cls, d)["articles"].values()
        out.append({"date": d, "articles": len(arts)})
    return out


def topic_of(a: dict) -> str:
    """The sport of an article as the model named it, tidied ("Field Hockey " -> "field hockey")."""
    s = " ".join(str(a.get("facts", {}).get("sport", "")).replace("_", " ").replace("-", " ").lower().split())
    return s if s and s != "unknown" else "other"


def topics(cls: str, day: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for a in store.load_day(cls, day)["articles"].values():
        if "facts" in a:
            groups.setdefault(topic_of(a), []).append({k: a.get(k) for k in KEEP if k in a})
    out = [{"name": k, "count": len(v), "articles": sorted(v, key=lambda x: (x["page"], x["id"]))}
           for k, v in groups.items()]
    return sorted(out, key=lambda t: (t["name"] == "other", -t["count"], t["name"]))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        cls = q.get("class", "sports")
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (UI_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/classes":
                return self._json(classes())
            if u.path == "/api/days":
                return self._json(days(cls))
            if u.path == "/api/topics":
                return self._json(topics(cls, q.get("date", "")))
            self._json({"error": "no such page"}, 404)
        except Exception as e:                      # show the problem in the page, do not crash the server
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):             # quieter terminal
        sys.stderr.write("%s %s\n" % (self.command, self.path.split("?")[0]))


def main():
    config.utf8_console()
    print(f"News UI on http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
