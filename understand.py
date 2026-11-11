"""
Step 4 - one more model call per article, after the facts are stored: what the news MEANS.

The model gets the article and the facts step 2 found (only those code found in the text) and writes:
    summary                2-3 sentences: who did what, where, when, the result
    why_it_matters         one sentence, only if the article shows it
    status                 happened / expected / mixed
    key_points             up to 5 facts worth remembering
    tags                   up to 8 lowercase search words
    answerable_questions   up to 5 questions this article answers (they help the search in step 5 match how
                           people ask)
Saved as "meaning" on the article in data/store/<class>/<date>.json. Prompt: prompts/<class>/8_meaning.txt.

Run (from graphrag/):
    python news/understand.py --date 2026-11-11                     every extracted article without a meaning yet
    python news/understand.py --date 2026-11-11 --article <id> --force
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import extract  # noqa: E402
import llm  # noqa: E402
import store  # noqa: E402


def obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props)}


STR = {"type": "string"}
MEANING_SCHEMA = obj({
    "summary": STR,
    "why_it_matters": STR,
    "status": {"type": "string", "enum": ["happened", "expected", "mixed"]},
    "key_points": {"type": "array", "items": STR, "maxItems": 5},
    "tags": {"type": "array", "items": STR, "maxItems": 8},
    "answerable_questions": {"type": "array", "items": STR, "maxItems": 5},
})


def facts_text(rec: dict) -> str:
    """The stored facts as short lines for the prompt, leaving out values code could not find in the article."""
    f = rec.get("facts", {})
    bad = {x["where"] for x in rec.get("checks", {}).get("not_in_text", [])}

    def ok(where: str) -> bool:
        return where not in bad

    lines = [f'sport: {f.get("sport", "")} | story type: {f.get("story_type", "")} | status: {f.get("status", "")}']
    if f.get("how_it_works"):
        lines.append(f'how the sport works: {f["how_it_works"]}')
    if f.get("winner") and ok("winner"):
        lines.append(f'winner: {f["winner"]}')
    event = [f.get(k, "") for k in ("competition", "discipline", "category", "stage", "venue", "city", "country")
             if ok(k)]
    if any(event):
        lines.append("event: " + ", ".join(x for x in event if x))
    if f.get("result_summary"):
        lines.append(f'result: {f["result_summary"]}')
    for i, r in enumerate(f.get("results", [])):
        if ok(f"results[{i}].value"):
            note = f' ({r["note"]})' if r.get("note") else ""
            lines.append(f'- {r.get("who", "")} {r.get("what", "")}: {r.get("value", "")}{note}'.replace("-  ", "- "))
    days = {d["field"]: d for d in rec.get("dates", [])}
    for i, d in enumerate(f.get("dates", [])):
        day = days.get(f"dates[{i}].date_text", {}).get("date", "")
        lines.append(f'date: {d.get("what", "")} = {d.get("date_text", "")}' + (f" ({day})" if day else ""))
    people = [f'{p["name"]} ({", ".join(x for x in (p.get("role"), p.get("affiliation")) if x)})'
              for i, p in enumerate(f.get("participants", [])) if ok(f"participants[{i}].name")]
    if people:
        lines.append("people and teams: " + "; ".join(people))
    for c in f.get("changes", []):
        lines.append(f'change: {c.get("who", "")} {c.get("change", "")} - {c.get("detail", "")}')
    for u in f.get("upcoming", []):
        lines.append(f'coming up: {u.get("what", "")} {u.get("date_text", "")} {u.get("where", "")}'.strip())
    return "\n".join(lines)


def understand_article(rec: dict, template: str) -> tuple[dict, dict]:
    prompt = llm.fill(template, headline=rec["headline"], paper=f'{rec["paper"]} ({rec["edition"]})',
                      paper_date=rec["paper_date"], page=rec["page"], dateline=extract.dateline(rec),
                      article=extract.article_for_model(rec), facts=facts_text(rec))
    r = llm.call(prompt, MEANING_SCHEMA, step="meaning", max_new_tokens=700)
    meaning = extract.tidy(r["parsed"], MEANING_SCHEMA) if r["parsed"] else llm.conform({}, MEANING_SCHEMA)
    meaning["ok"] = r["parsed"] is not None
    log = {"article": rec["id"], "prompt_chars": len(prompt), "at": datetime.now().isoformat(timespec="seconds"),
           **r["log"]}
    return meaning, log


def run(day: str, cls: str = "sports", article: str | None = None, limit: int | None = None,
        force: bool = False) -> None:
    template = (config.PROMPTS / cls / "8_meaning.txt").read_text(encoding="utf-8")
    data = store.load_day(cls, day)
    recs = [r for r in data["articles"].values() if "facts" in r and (not article or r["id"] == article)]
    todo = [r for r in recs if force or not r.get("meaning")]
    if limit:
        todo = todo[:limit]
    print(f"{len(recs)} extracted {cls} articles on {day}; {len(todo)} to explain with {llm.MODEL_NAME}")
    for n, rec in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {rec['id']}  {rec['headline'][:80]}")
        meaning, log = understand_article(rec, template)
        rec["meaning"] = meaning
        rec.setdefault("runs", {})["meaning"] = {"model": llm.MODEL_NAME, "at": log["at"], "seconds": log["seconds"]}
        store.save_day(data)
        store.log_calls(f"{day}_{cls}_meaning", [log])
        print(f"    {log['seconds']}s  {'ok' if meaning['ok'] else 'FAILED (no JSON)'}")
        print(f"    {meaning['summary'][:300]}")
    print(f"\nSaved {store.day_path(cls, day)}")


def main():
    config.utf8_console()
    ap = argparse.ArgumentParser(description="Step 4: what each stored article means")
    ap.add_argument("--date", required=True, help="the paper's date, YYYY-MM-DD")
    ap.add_argument("--class", dest="cls", default="sports")
    ap.add_argument("--article")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.date, a.cls, a.article, a.limit, a.force)


if __name__ == "__main__":
    main()
