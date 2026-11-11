"""
Paths and settings shared by every step of the news pipeline. Everything lives inside graphrag/news/.

    papers/                       the newspaper PDFs (not committed)
    prompts/<class>/              the prompts of one class (sports, business ...) - a class is ready once it has
                                  extract_steps.json
    data/articles/<date>/         step 1: the articles cut out of each paper, plus a .review.txt to check by eye
    data/store/<class>/<date>.json   steps 2-4: one file per class and day: facts, checks and meaning per article
    data/index/<class>/<date>.json   step 5: the search vectors of that day's articles
    data/logs/                    every model call (prompt size, raw reply, seconds)
    models/                       the embedding model (python news/download_models.py)
"""

import os
from pathlib import Path

NEWS = Path(__file__).resolve().parent          # graphrag/news
REPO = NEWS.parent                               # graphrag
PAPERS = NEWS / "papers"
PROMPTS = NEWS / "prompts"
DATA = NEWS / "data"
ARTICLES = DATA / "articles"
STORE = DATA / "store"
INDEX = DATA / "index"
LOGS = DATA / "logs"
MODELS = NEWS / "models"

# The page labels printed at the top of The Statesman's pages (letters spaced out, e.g. "S P O RT S") -> class.
# Matched after removing spaces. Page 1 has no label and is "front"; a page of another size (an advert) is skipped.
PAGE_LABELS = {
    "SPORTS": "sports", "SPORT": "sports",
    "BUSINESS": "business",
    "WORLD": "world",
    "INDIA": "india", "NATION": "india", "INDIA/CAPITAL": "india",
    "CAPITAL": "capital", "CAPITAL/INDIA": "capital",
    "EDITORIAL": "editorial", "OPINION": "opinion", "PERSPECTIVE": "opinion",
    "STATES": "states",
}

# Text sent to the model per article; longer articles are cut (a small model loses track anyway)
MAX_ARTICLE_CHARS = int(os.getenv("NEWS_MAX_ARTICLE_CHARS", "6000"))

# Embedding model for search (step 5). Downloaded to models/<name>; if missing, search is keyword-only.
EMBED_MODEL_ID = os.getenv("NEWS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_MODEL_DIR = MODELS / EMBED_MODEL_ID.split("/")[-1]


def utf8_console() -> None:
    """Print UTF-8 (₹, ’, é) on the Windows console instead of crashing on it."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def classes_with_prompts() -> list[str]:
    """The classes that have their prompt set ready (prompts/<class>/extract_steps.json)."""
    if not PROMPTS.exists():
        return []
    return sorted(p.name for p in PROMPTS.iterdir() if (p / "extract_steps.json").exists())
