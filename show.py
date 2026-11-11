"""
Show what is stored for a class and day, in plain text: per article the meaning, the facts, the resolved dates
and the check results (values the model wrote that are not in the article).

Run (from graphrag/):
    python news/show.py --date 2026-11-11                    every Sports article of that day
    python news/show.py --date 2026-11-11 --article <id>     one article, with its full text
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import store  # noqa: E402


def show(rec: dict, full_text: bool = False) -> None:
    f, m, c = rec.get("facts", {}), rec.get("meaning") or {}, rec.get("checks", {})
    print("=" * 100)
    print(f'{rec["id"]}   {rec["paper"]} ({rec["edition"]}) {rec["paper_date"]}, page {rec["page"]}')
    print(rec["headline"])
    print(f'{rec.get("agency") or "-"} | {rec.get("dateline_place") or "-"} | {rec.get("dateline_date") or "no dateline date"}')
    if m:
        print(f'\nMEANING ({m.get("status", "")})\n  {m.get("summary", "")}')
        if m.get("why_it_matters"):
            print(f'  Why it matters: {m["why_it_matters"]}')
        for p in m.get("key_points", []):
            print(f"  * {p}")
        print(f'  tags: {", ".join(m.get("tags", []))}')
        for q in m.get("answerable_questions", []):
            print(f"  ? {q}")
    if f:
        print(f'\nFACTS  {f.get("sport", "")} / {f.get("story_type", "")} / {f.get("status", "")}')
        print(f'  gist: {f.get("gist", "")}')
        print(f'  the sport ({f.get("contest_type", "")}): {f.get("how_it_works", "") or "-"}')
        if f.get("result_words"):
            print(f'  counting words in the article: {", ".join(f["result_words"])}')
        if f.get("winner"):
            print(f'  winner: {f["winner"]}')
        keys = ("competition", "discipline", "category", "stage", "venue", "city", "country")
        event = ", ".join(x for x in (f.get(k, "") for k in keys) if x)
        print(f"  event: {event or '-'}")
        if f.get("result_summary"):
            print(f'  result: {f["result_summary"]}')
        for r in f.get("results", []):
            print(f'  result  {r.get("who", "")} | {r.get("what", "")} | {r.get("value", "")}'
                  + (f' | {r["note"]}' if r.get("note") else ""))
        for p in f.get("participants", []):
            print(f'  {p.get("kind", ""):<12} {p.get("name", "")} - {p.get("role", "")}'
                  + (f' ({p["affiliation"]})' if p.get("affiliation") else ""))
        for ch in f.get("changes", []):
            print(f'  change  {ch.get("who", "")}: {ch.get("change", "")} - {ch.get("detail", "")}')
        for q in f.get("quotes", []):
            print(f'  quote   {q.get("speaker", "")}: "{q.get("quote", "")}"')
        for u in f.get("upcoming", []):
            print(f'  next    {u.get("what", "")} | {u.get("date_text", "")} | {u.get("where", "")}')
    for d in rec.get("dates", []):
        day = d["date"] + (f' to {d["end"]}' if d.get("end") else "") if d["date"] else "?"
        print(f'  date    "{d["text"]}" -> {day} ({d["how"]})  [{d.get("what", "")}]')
    if c.get("not_in_text") or c.get("failed_steps") or c.get("cut_off_steps"):
        print("\nCHECKS")
        for x in c.get("not_in_text", []):
            print(f'  not in the article: {x["where"]} = "{x["value"]}"')
        if c.get("failed_steps"):
            print(f'  steps with no JSON: {", ".join(c["failed_steps"])}')
        if c.get("cut_off_steps"):
            print(f'  steps cut off (max tokens or a loop): {", ".join(c["cut_off_steps"])}')
    if full_text:
        print("\nTEXT\n" + rec.get("text", ""))
    print()


def main():
    config.utf8_console()
    ap = argparse.ArgumentParser(description="Show stored articles")
    ap.add_argument("--date", required=True)
    ap.add_argument("--class", dest="cls", default="sports")
    ap.add_argument("--article")
    a = ap.parse_args()
    data = store.load_day(a.cls, a.date)
    recs = [r for r in data["articles"].values() if not a.article or r["id"] == a.article]
    if not recs:
        sys.exit(f"Nothing stored for {a.cls} on {a.date}. Stored days: {store.days(a.cls)}")
    for rec in sorted(recs, key=lambda r: r["id"]):
        show(rec, full_text=bool(a.article))
    print(f"{len(recs)} articles from {store.day_path(a.cls, a.date)}")


if __name__ == "__main__":
    main()
