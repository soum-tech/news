"""
Steps 2-3 - take the facts out of each article of one class, check them, and store them.

Step 2 (model): the calls listed in prompts/<class>/extract_steps.json, in order, one article at a time. For Sports:
    1 identify       the sport (named by the model in its own words - there is no list of sports), story type,
                     country, one-sentence gist
    2 sport_profile  the model explains the sport: how it works and how a winner is decided, what kind of contest
                     this article reports (two teams / two players / a ranked field / no contest), and the
                     counting words the article uses
    3 participants   players, teams, coaches, officials, organisations, with their role
    4 event          competition, discipline, category, stage, venue, city, country, the dates (as written)
    5 outcome        status, one-sentence result, the winner
    6 figures        every score / time / distance / placing / vote (copied exactly), each with a note for
                     records, personal bests or qualification
    7 changes_next   squad changes, injuries, elections, qualification; quotes; what comes next
Each call is small on purpose: a 0.8B model asked for everything at once loses track.
Passing facts forward: every text fact a call returns can be used by a later prompt under its own key
(<sport>, <gist>, <how_it_works>, <result_words> ...). So calls 4-6 read the article with the model's own
explanation of the sport in front of them, whatever the sport is.
Hints: the "hints" table in extract_steps.json adds one line to calls 4 and 6, chosen by the kind of contest
call 2 found - never by the sport's name.

Step 3 (code, no model):
    - tidy: empty items and exact repeats dropped
    - check: every name / value listed under "check_in_text" is looked up in the article; misses are listed in
      checks.not_in_text (a sign the model guessed or misread). Nothing is deleted, so you can judge.
    - dates: every date the model copied ("Tuesday", "November 13 and 15") becomes a real day, counted from the
      article's dateline (textutil.resolve_date). Code does this, not the model.
    - store: data/store/<class>/<paper date>.json (store.py), saved after every article.

Run (from graphrag/):
    python news/extract.py --date 2026-11-11                        all Sports articles of that day not done yet
    python news/extract.py --date 2026-11-11 --limit 1              just the next one (to time the model)
    python news/extract.py --date 2026-11-11 --article <id> --force    one article again
Every model call is logged in data/logs/<date>_<class>_extract.jsonl (raw reply, tokens, seconds).
"""

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import llm  # noqa: E402
import store  # noqa: E402
import textutil  # noqa: E402

# fields copied from the step-1 article into the store
ARTICLE_FIELDS = ("id", "paper", "edition", "paper_date", "page", "class", "headline", "kicker", "agency",
                  "dateline_place", "dateline_date", "base_date", "text")


# ---------------- loading ----------------
def load_steps(cls: str) -> dict:
    p = config.PROMPTS / cls / "extract_steps.json"
    if not p.exists():
        sys.exit(f"No prompt set for class {cls!r} ({p} is missing). Classes ready: {config.classes_with_prompts()}")
    spec = json.loads(p.read_text(encoding="utf-8"))
    keys = [k for s in spec["steps"] for k in s["schema"]["properties"]]
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        sys.exit(f"{p}: these keys appear in more than one step, so they would overwrite each other: {repeated}")
    for s in spec["steps"]:
        s["template"] = (config.PROMPTS / cls / s["prompt"]).read_text(encoding="utf-8")
    hints = spec.get("hints") or {}
    if hints and hints.get("by") not in keys:
        sys.exit(f'{p}: hints are chosen by "{hints.get("by")}", but no step returns that key')
    return spec


def hint_names(spec: dict) -> set[str]:
    """Every hint placeholder the class defines (for Sports: event_hint, results_hint)."""
    return {name for row in (spec.get("hints") or {}).get("table", {}).values() for name in row}


def context(facts: dict, spec: dict) -> dict:
    """What later prompts may use from the calls done so far: every text fact under its own key (a list of
    words joined with commas), the people found, and the hint lines for the value the hints are chosen by."""
    out = {}
    for k, v in facts.items():
        if isinstance(v, str):
            out[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[k] = ", ".join(v) or "none found"
    out["participants_list"] = participants_list(facts)
    hints = spec.get("hints") or {}
    row = hints.get("table", {}).get(facts.get(hints.get("by", ""), ""), {})
    out.update({name: row.get(name, "") for name in hint_names(spec)})
    return out


def load_articles(day: str, cls: str, paper: str | None = None) -> list[dict]:
    """The step-1 articles of one class on one day (every paper of that day, or one paper slug)."""
    folder = config.ARTICLES / day
    files = sorted(folder.glob("*.json")) if folder.exists() else []
    if not files:
        sys.exit(f"No articles for {day} in {folder}: run split_pdf.py first")
    out = []
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        out += [a for a in r["articles"] if a["class"] == cls and (not paper or a["id"].startswith(paper))]
    return out


# ---------------- prompt pieces ----------------
def dateline(a: dict) -> str:
    parts = [a.get("agency"), a.get("dateline_place"), a.get("dateline_date")]
    return ", ".join(p for p in parts if p) or "not given"


def article_for_model(a: dict) -> str:
    text = a["text"]
    if len(text) <= config.MAX_ARTICLE_CHARS:
        return text
    cut = text[: config.MAX_ARTICLE_CHARS]
    return cut[: cut.rfind(". ") + 1 or len(cut)] + " [... rest of the article cut ...]"


def participants_list(facts: dict) -> str:
    people = [f'{p["name"]} ({p["role"]})' if p.get("role") else p["name"]
              for p in facts.get("participants", []) if p.get("name")]
    return "; ".join(people) or "none listed"


# ---------------- step 3: tidy, check, dates ----------------
def tidy(v, schema: dict):
    """Strip strings; drop list items whose free-text fields are all empty, and exact repeats."""
    t = schema.get("type")
    if t == "object":
        v = v if isinstance(v, dict) else {}
        return {k: tidy(v.get(k), s) for k, s in schema["properties"].items()}
    if t == "array":
        item_schema = schema["items"]
        free = [k for k, s in item_schema.get("properties", {}).items() if "enum" not in s]
        out, seen = [], set()
        for x in v or []:
            x = tidy(x, item_schema)
            if isinstance(x, dict) and free and not any(x.get(k) for k in free):
                continue
            if x in ("", None):
                continue
            key = json.dumps(x, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                out.append(x)
        return out
    return " ".join(str(v).split()) if isinstance(v, str) else v


def walk(facts: dict, path: str):
    """("where", the dict holding the value, field) for a path like "results[].value", "result_words[]" (a list
    of words) or "venue"."""
    if path.endswith("[]"):
        key = path[:-2]
        for i, item in enumerate(facts.get(key) or []):
            if isinstance(item, str):
                yield f"{key}[{i}]", {"_": item}, "_"
    elif "[]." in path:
        lst, field = path.split("[].", 1)
        for i, item in enumerate(facts.get(lst) or []):
            if isinstance(item, dict):
                yield f"{lst}[{i}].{field}", item, field
    else:
        yield path, facts, path


def not_in_text(facts: dict, text: str, paths: list[str]) -> list[dict]:
    text_norm = textutil.norm(text)
    out = []
    for path in paths:
        for where, holder, field in walk(facts, path):
            v = holder.get(field, "")
            if isinstance(v, str) and v and not textutil.found_in(v, text_norm):
                out.append({"where": where, "value": v})
    return out


def resolve_dates(facts: dict, base: date, spec_dates: dict) -> list[dict]:
    """Every date the model copied out of the article, as a real day."""
    upcoming = facts.get("story_type") == "preview" or facts.get("status") == "upcoming"
    out = []
    for path, prefer in spec_dates.items():
        if prefer == "auto":
            prefer = "future" if upcoming else "past"
        for where, holder, field in walk(facts, path):
            written = holder.get(field, "")
            if written:
                r = textutil.resolve_date(written, base, prefer)
                out.append({"field": where, "what": holder.get("what", ""), "text": written, **r})
    return out


# ---------------- one article ----------------
def extract_article(a: dict, spec: dict) -> tuple[dict, list[dict]]:
    """Run every step on one article -> (store record, call logs)."""
    ctx = {k: "" for s in spec["steps"] for k in s["schema"]["properties"]}     # facts not found yet are blank
    ctx.update({"headline": a["headline"], "dateline": dateline(a), "paper_date": a["paper_date"],
                "article": article_for_model(a)})
    facts, logs, failed, cut = {}, [], [], []
    ctx.update(context(facts, spec))
    t0 = time.time()
    for s in spec["steps"]:
        prompt = llm.fill(s["template"], **ctx)
        r = llm.call(prompt, s["schema"], step=s["id"], max_new_tokens=s.get("max_new_tokens"))
        logs.append({"article": a["id"], "prompt_chars": len(prompt), "at": datetime.now().isoformat(timespec="seconds"),
                     **r["log"]})
        if r["parsed"] is None:
            failed.append(s["id"])
            parsed = llm.conform({}, s["schema"])              # empty values, right shape
        else:
            parsed = tidy(r["parsed"], s["schema"])
        if r["log"]["truncated"]:
            cut.append(s["id"])
        for k in spec.get("lowercase", []):                    # "Cricket" and "cricket" are the same sport
            if isinstance(parsed.get(k), str):
                parsed[k] = parsed[k].lower()
        facts.update(parsed)
        ctx.update(context(facts, spec))
        flag = "ok" if r["parsed"] is not None else "FAILED (no JSON)"
        print(f'    {s["id"]:<13} {r["log"]["seconds"]:>6.1f}s  {r["log"]["output_tokens"]:>4} tokens  {flag}'
              + ("  (cut off)" if r["log"]["truncated"] else ""))
    base = date.fromisoformat(a["base_date"])
    record = {k: a.get(k, "") for k in ARTICLE_FIELDS}
    record.update({
        "facts": facts,
        "dates": resolve_dates(facts, base, spec.get("dates", {})),
        "checks": {"not_in_text": not_in_text(facts, f'{a["headline"]} {a["text"]}', spec.get("check_in_text", [])),
                   "failed_steps": failed, "cut_off_steps": cut},
        "runs": {"extract": {"model": llm.MODEL_NAME, "at": datetime.now().isoformat(timespec="seconds"),
                             "seconds": round(time.time() - t0, 1)}},
    })
    return record, logs


# ---------------- a day ----------------
def run(day: str, cls: str = "sports", paper: str | None = None, article: str | None = None,
        limit: int | None = None, force: bool = False) -> None:
    spec = load_steps(cls)
    arts = load_articles(day, cls, paper)
    if article:
        arts = [a for a in arts if a["id"] == article]
        if not arts:
            sys.exit(f"No {cls} article {article!r} on {day}")
    data = store.load_day(cls, day)
    todo = [a for a in arts if force or "facts" not in data["articles"].get(a["id"], {})]
    if limit:
        todo = todo[:limit]
    print(f"{len(arts)} {cls} articles on {day}; extracting {len(todo)} with {llm.MODEL_NAME} "
          f"({len(spec['steps'])} calls each)")
    for n, a in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {a['id']}  p{a['page']}  {a['headline'][:80]}")
        record, logs = extract_article(a, spec)
        data["articles"][a["id"]] = record                     # replaces any earlier record, meaning included
        store.save_day(data)
        store.log_calls(f"{day}_{cls}_extract", logs)
        f, c = record["facts"], record["checks"]
        print(f"    -> {f.get('sport')} / {f.get('story_type')} / {f.get('contest_type')}: {f.get('gist', '')[:100]}")
        print(f"    -> how it works: {f.get('how_it_works', '')[:140]}")
        print(f"    -> {len(f.get('participants', []))} participants, {len(f.get('results', []))} results, "
              f"{len(record['dates'])} dates, {len(c['not_in_text'])} values not found in the text, "
              f"{record['runs']['extract']['seconds']}s")
    print(f"\nSaved {store.day_path(cls, day)}")


def main():
    config.utf8_console()
    ap = argparse.ArgumentParser(description="Steps 2-3: facts out of each article, checked and stored")
    ap.add_argument("--date", required=True, help="the paper's date, YYYY-MM-DD")
    ap.add_argument("--class", dest="cls", default="sports")
    ap.add_argument("--paper", help="only this paper's articles (id prefix, e.g. statesman-delhi)")
    ap.add_argument("--article", help="only this article id")
    ap.add_argument("--limit", type=int, help="at most this many articles")
    ap.add_argument("--force", action="store_true", help="redo articles that are already stored")
    a = ap.parse_args()
    run(a.date, a.cls, a.paper, a.article, a.limit, a.force)


if __name__ == "__main__":
    main()
