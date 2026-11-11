"""
Step 5a - search vectors for one class and day, so a question can find an article by meaning ("who is leaving
Manchester United?" finds "Manchester expected to sack Carrick").

Each article is turned into one search text (headline, gist, the meaning, its tags and questions, the main names,
the competition) and embedded with a small sentence-embedding model (default BAAI/bge-small-en-v1.5, about
130 MB, CPU is fine). Saved to data/index/<class>/<date>.json.

The model must be downloaded first: python news/download_models.py
Without it, ask.py still works with keyword search only.

Run (from graphrag/):
    python news/search_index.py --date 2026-11-11
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import store  # noqa: E402

# bge models want this in front of a QUESTION (not in front of the articles)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_model = {}


def load_embedder():
    """The embedding model, or None (with the reason printed) if it cannot be used."""
    if "m" in _model:
        return _model["m"]
    _model["m"] = None
    if not (config.EMBED_MODEL_DIR / "config.json").exists():
        print(f"(no embedding model in {config.EMBED_MODEL_DIR}: run python news/download_models.py; "
              f"using keyword search only)")
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("(sentence-transformers is not installed: pip install -r news/requirements.txt; keyword search only)")
        return None
    _model["m"] = SentenceTransformer(str(config.EMBED_MODEL_DIR), device="cpu")
    return _model["m"]


def search_text(rec: dict) -> str:
    f, m = rec.get("facts", {}), rec.get("meaning") or {}
    parts = [rec["headline"], f.get("gist", ""), m.get("summary", ""), m.get("why_it_matters", ""),
             " ".join(m.get("key_points", [])), " ".join(m.get("tags", [])), " ".join(m.get("answerable_questions", [])),
             " ".join(p.get("name", "") for p in f.get("participants", [])[:8]),
             " ".join(f.get(k, "").replace("_", " ") for k in ("sport", "competition", "discipline", "category",
                                                                 "stage", "venue", "city"))]
    if not m:                                   # no meaning yet: fall back to the start of the article
        parts.append(rec.get("text", "")[:800])
    return " ".join(" ".join(p.split()) for p in parts if p)


def embed(texts: list[str], query: bool = False):
    model = load_embedder()
    if model is None:
        return None
    if query:
        texts = [QUERY_PREFIX + t for t in texts]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def index_path(cls: str, day: str):
    return config.INDEX / cls / f"{day}.json"


def build(day: str, cls: str = "sports") -> None:
    data = store.load_day(cls, day)
    recs = sorted(data["articles"].values(), key=lambda r: r["id"])
    if not recs:
        print(f"Nothing stored for {cls} on {day}")
        return
    vectors = embed([search_text(r) for r in recs])
    if vectors is None:
        return
    out = {"class": cls, "date": day, "model": config.EMBED_MODEL_ID,
           "made_at": datetime.now().isoformat(timespec="seconds"),
           "vectors": {r["id"]: [round(float(x), 5) for x in v] for r, v in zip(recs, vectors)}}
    store.write_json(index_path(cls, day), out)
    without = sum(1 for r in recs if not r.get("meaning"))
    print(f"Indexed {len(recs)} {cls} articles of {day} -> {index_path(cls, day)}"
          + (f" ({without} without a meaning yet: run understand.py, then index again)" if without else ""))


def main():
    config.utf8_console()
    ap = argparse.ArgumentParser(description="Step 5a: search vectors for one class and day")
    ap.add_argument("--date", required=True)
    ap.add_argument("--class", dest="cls", default="sports")
    a = ap.parse_args()
    build(a.date, a.cls)


if __name__ == "__main__":
    main()
