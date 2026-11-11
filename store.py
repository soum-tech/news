"""
The store: one JSON file per class and day, data/store/<class>/<YYYY-MM-DD>.json (the paper's date).

    {"class": "sports", "date": "2026-11-11",
     "articles": {"<article id>": {
         "id", "paper", "edition", "paper_date", "page", "class", "headline", "kicker",
         "agency", "dateline_place", "dateline_date", "text",          <- step 1 (split_pdf.py)
         "facts":   {sport, story_type, gist, how_it_works, contest_type, participants, competition, ... },
                                                                       <- step 2 (extract.py)
         "dates":   [{"field", "text", "date", "end", "how"}],         <- step 3: every date resolved to a day
         "checks":  {"not_in_text": [...], "failed_steps": [...], "cut_off_steps": [...]},
         "meaning": {summary, why_it_matters, status, key_points, tags, answerable_questions},   <- step 4
         "runs":    {"extract": {...}, "meaning": {...}}               <- model, time, seconds
     }}}

Files are written to a temporary file first and then swapped in, so a run stopped half-way never leaves a
broken file; every step saves after each article, so a stopped run continues where it left off.
"""

import json
import os
from datetime import date

import config


def day_path(cls: str, day: str):
    return config.STORE / cls / f"{day}.json"


def load_day(cls: str, day: str) -> dict:
    p = day_path(cls, day)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"class": cls, "date": day, "articles": {}}


def save_day(data: dict) -> None:
    p = day_path(data["class"], data["date"])
    write_json(p, data)


def write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def days(cls: str) -> list[str]:
    """Every day stored for a class, oldest first."""
    folder = config.STORE / cls
    return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []


def all_articles(cls: str) -> list[dict]:
    """Every stored article of a class, all days."""
    out = []
    for d in days(cls):
        out.extend(load_day(cls, d)["articles"].values())
    return out


def article_days(a: dict) -> set[str]:
    """The days an article is about: the paper's date, its dateline date, and every date the article mentions.
    A question about "10 November" finds the paper of 11 November, whose stories are datelined 10 November."""
    out = {a.get("paper_date", ""), a.get("dateline_date", "")}
    for d in a.get("dates", []):
        out.update({d.get("date", ""), d.get("end", "")})
        if d.get("date") and d.get("end"):                 # every day of a range, e.g. a tournament 19-28 November
            first, last = date.fromisoformat(d["date"]), date.fromisoformat(d["end"])
            if 0 < (last - first).days <= 31:
                out.update(date.fromordinal(i).isoformat() for i in range(first.toordinal(), last.toordinal() + 1))
    out.discard("")
    return out


def log_calls(name: str, records: list[dict]) -> None:
    """Append model-call logs to data/logs/<name>.jsonl."""
    if not records:
        return
    p = config.LOGS / f"{name}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
