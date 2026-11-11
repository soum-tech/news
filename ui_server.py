"""
The news UI server: serves news/ui/index.html and a small JSON API over the stored data (no extra packages).

    python news/ui_server.py                 then open http://localhost:8001

API (all GET unless noted):
    /api/classes                              classes that have stored days
    /api/days?class=sports                    stored days with article counts
    /api/articles?class=sports&date=2026-11-11   article list for one day (summary fields only)
    /api/article?class=sports&date=...&id=...    one article, complete
    /api/search?class=sports&q=...&date=...   ranked articles, keyword + meaning, no model call (fast)
    POST /api/ask  {"question", "class", "date"}  answer from the top articles (runs the model: slow on CPU)
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import store  # noqa: E402
import textutil  # noqa: E402

UI_DIR = config.NEWS / "ui"
PORT = 8001
_ask_lock = None


def classes() -> list[str]:
    return sorted(p.name for p in config.STORE.iterdir() if p.is_dir()) if config.STORE.exists() else []


def days(cls: str) -> list[dict]:
    out = []
    for d in store.days(cls):
        arts = store.load_day(cls, d)["articles"].values()
        out.append({"date": d, "articles": len(arts), "with_meaning": sum(1 for a in arts if a.get("meaning"))})
    return out


def summary(a: dict) -> dict:
    f, m, c = a.get("facts", {}), a.get("meaning") or {}, a.get("checks", {})
    return {"id": a["id"], "page": a["page"], "headline": a["headline"], "paper": a["paper"],
            "edition": a["edition"], "paper_date": a["paper_date"], "sport": f.get("sport", ""),
            "story_type": f.get("story_type", ""), "status": f.get("status", ""), "gist": f.get("gist", ""),
            "summary": m.get("summary", ""), "tags": m.get("tags", []), "meaning_status": m.get("status", ""),
            "winner": f.get("winner", ""), "suggestions": len(a.get("suggestions", [])), "dates": [d.get("date", "") for d in a.get("dates", []) if d.get("date")],
            "not_in_text": len(c.get("not_in_text", [])), "failed_steps": c.get("failed_steps", [])}


def articles(cls: str, day: str) -> list[dict]:
    recs = store.load_day(cls, day)["articles"].values()
    return sorted((summary(a) for a in recs), key=lambda s: (s["page"], s["id"]))


def article(cls: str, day: str, art_id: str) -> dict | None:
    return store.load_day(cls, day)["articles"].get(art_id)


def search(cls: str, q: str, day: str | None) -> list[dict]:
    import ask
    recs = [r for r in store.all_articles(cls) if r.get("facts")]
    if day:
        recs = [r for r in recs if day in store.article_days(r)]
    if not q.strip():
        return [summary(r) | {"score": 0.0} for r in sorted(recs, key=lambda r: (r["paper_date"], r["page"]))]
    ranked = ask.rank(q, recs, cls)
    return [summary(r) | {"score": round(float(s), 3)} for s, r in ranked if s > 0][:30]


def ask_question(body: dict) -> dict:
    """Runs the model over the best matches. The first call loads the model, which takes a while."""
    import ask
    import threading
    global _ask_lock
    if _ask_lock is None:
        _ask_lock = threading.Lock()
    question = (body.get("question") or "").strip()
    cls = body.get("class") or "sports"
    if not question:
        return {"error": "empty question"}
    day = body.get("date") or ""
    recs = [r for r in store.all_articles(cls) if r.get("facts")]
    asked = textutil.find_date(question, __import__("datetime").date.today())
    if day:
        asked = __import__("datetime").date.fromisoformat(day)
    if asked:
        recs = [r for r in recs if asked.isoformat() in store.article_days(r)]
    if not recs:
        return {"found": "no", "answer": "No stored news for that day or class.", "sources": []}
    top = [r for s, r in ask.rank(question, recs, cls)[:3]]
    with _ask_lock:                               # one question at a time: the model is shared
        res = ask.answer(question, asked.isoformat() if asked else "not given", top, cls)
    return {"found": res["found"], "answer": res["answer"], "evidence": res["evidence"],
            "evidence_in_article": res["evidence_in_article"], "seconds": res["log"]["seconds"],
            "sources": [summary(s) for s in res["sources"]], "given": [summary(r) for r in top]}


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
            if u.path == "/api/articles":
                return self._json(articles(cls, q.get("date", "")))
            if u.path == "/api/article":
                a = article(cls, q.get("date", ""), q.get("id", ""))
                return self._json(a, 200) if a else self._json({"error": "not found"}, 404)
            if u.path == "/api/search":
                return self._json(search(cls, q.get("q", ""), q.get("date") or None))
            self._json({"error": "no such page"}, 404)
        except Exception as e:                      # show the problem in the page, do not crash the server
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        if urlparse(self.path).path != "/api/ask":
            return self._json({"error": "no such page"}, 404)
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            self._json(ask_question(body))
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):             # quieter terminal
        sys.stderr.write("%s %s\n" % (self.command, self.path.split("?")[0]))


def main():
    config.utf8_console()
    print(f"News UI on http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
