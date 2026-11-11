"""
Step 4b - suggestions: what a reader may want to know or follow, written from the stored facts of each article
and shown in the UI as grouped cards, the same way the medical suggestions were.

One model call per article (prompts/<class>/10_suggestions.txt) returns up to 10 suggestions:
    category   summary, understand_terms, background, people_to_know, what_next, follow_the_story, or a new one
    title, detail
    based_on   the fact that triggered it, "key: value" copied from the facts or the article
    verify     false when it comes straight from the article, true when it uses the model's general knowledge

Code then checks every number written in a suggestion against the article and its facts. A suggestion with a
number that is not there is marked verify = true and lists the numbers in "unmatched_numbers", so nothing
unchecked looks like it came from the article.

Saved as "suggestions" on the article in data/store/<class>/<date>.json.

Run (from graphrag/ or the repo root):
    python news/suggest.py --date 2026-11-11                  every extracted article without suggestions yet
    python news/suggest.py --date 2026-11-11 --article <id> --force
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import extract  # noqa: E402
import llm  # noqa: E402
import store  # noqa: E402
import textutil  # noqa: E402
import understand  # noqa: E402

STR = {"type": "string"}
SUGGESTIONS_SCHEMA = understand.obj({"suggestions": {"type": "array", "maxItems": 10, "items": understand.obj({
    "category": STR, "title": STR, "detail": STR, "based_on": STR,
    # an unknown value becomes the last one, "yes": when in doubt, verify
    "verify": {"type": "string", "enum": ["no", "yes"]}})}})
NUMBER = re.compile(r"\d[\d,.:/-]*\d|\d")


def unmatched_numbers(s: dict, reference: str) -> list[str]:
    """Numbers written in a suggestion that appear nowhere in the article text or the facts."""
    ref_digits = re.sub(r"[^0-9]+", " ", reference)
    ref_set = set(ref_digits.split()) | {w.replace(",", "") for w in NUMBER.findall(reference)}
    out = []
    for n in NUMBER.findall(f'{s["title"]} {s["detail"]}'):
        parts = [p for p in re.split(r"[^0-9]+", n) if p]
        if parts and not all(p in ref_set for p in parts):
            out.append(n)
    return out


def suggest_article(rec: dict, template: str) -> tuple[list[dict], dict]:
    facts = understand.facts_text(rec)
    prompt = llm.fill(template, headline=rec["headline"], paper=f'{rec["paper"]} ({rec["edition"]})',
                      paper_date=rec["paper_date"], dateline=extract.dateline(rec),
                      article=extract.article_for_model(rec), facts=facts)
    r = llm.call(prompt, SUGGESTIONS_SCHEMA, step="suggestions", max_new_tokens=1000)
    reference = f'{rec["headline"]} {rec["text"]} {facts}'
    items = []
    for s in (extract.tidy(r["parsed"], SUGGESTIONS_SCHEMA)["suggestions"] if r["parsed"] else []):
        s["category"] = re.sub(r"[^a-z0-9]+", "_", s["category"].lower()).strip("_") or "other"
        s["verify"] = s["verify"] != "no"
        bad = unmatched_numbers(s, reference)
        if bad:
            s["verify"], s["unmatched_numbers"] = True, bad
        items.append(s)
    log = {"article": rec["id"], "prompt_chars": len(prompt), "at": datetime.now().isoformat(timespec="seconds"),
           **r["log"]}
    return items, log


def run(day: str, cls: str = "sports", article: str | None = None, limit: int | None = None,
        force: bool = False) -> None:
    template = (config.PROMPTS / cls / "10_suggestions.txt").read_text(encoding="utf-8")
    data = store.load_day(cls, day)
    recs = [r for r in data["articles"].values() if "facts" in r and (not article or r["id"] == article)]
    todo = [r for r in recs if force or "suggestions" not in r]
    if limit:
        todo = todo[:limit]
    print(f"{len(recs)} extracted {cls} articles on {day}; {len(todo)} to get suggestions with {llm.MODEL_NAME}")
    for n, rec in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {rec['id']}  {rec['headline'][:80]}")
        items, log = suggest_article(rec, template)
        rec["suggestions"] = items
        rec.setdefault("runs", {})["suggestions"] = {"model": llm.MODEL_NAME, "at": log["at"], "seconds": log["seconds"]}
        store.save_day(data)
        store.log_calls(f"{day}_{cls}_suggestions", [log])
        print(f"    {log['seconds']}s  {'ok' if log['ok'] else 'FAILED (no JSON)'}  {len(items)} suggestions, "
              f"{sum(1 for s in items if s['verify'])} to verify")
    print(f"\nSaved {store.day_path(cls, day)}")


def main():
    config.utf8_console()
    ap = argparse.ArgumentParser(description="Step 4b: suggestions for each stored article")
    ap.add_argument("--date", required=True, help="the paper's date, YYYY-MM-DD")
    ap.add_argument("--class", dest="cls", default="sports")
    ap.add_argument("--article")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.date, a.cls, a.article, a.limit, a.force)


if __name__ == "__main__":
    main()
