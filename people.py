"""
Step 4c - the people in the news: who they are, their sport, and their past and present record.

For each person named in an article (kind "person", a full name, at most 4 per article, in the order the article
gives them):
  1. Code looks the name up on Wikipedia (free, no key): search "<name> <sport>", take the first page whose title
     matches the name AND whose opening text sounds like a sports person (or mentions the article's sport).
     No match -> no profile is made. Nothing is guessed.
  2. The model (prompts/<class>/11_person_profile.txt) reads only that page's opening text and writes the
     sport, a short background, career records and the current status. It also says whether the page is the
     same person; "no" drops the profile.
  3. Code checks every number in the profile against the page text; a number that is not there marks the
     profile "verify" (profile.unmatched_numbers).
  4. What the article itself says about the person (role, team, figures like scores) is attached too, from the
     facts already stored, so the card shows both.

Saved as "people" on the article in data/store/<class>/<date>.json. Lookups are cached in
data/people_cache.json, so a person appearing in many articles is looked up once. Needs internet for the lookup.
The profile can be out of date: its card says "Source: Wikipedia, fetched <date>".

Run (from the repo root):
    python people.py --date 2026-11-11                  every extracted article without people yet
    python people.py --date 2026-11-11 --article <id> --force
    python people.py --lookup "Manu Bhaker" --sport shooting      just test the Wikipedia lookup
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import extract  # noqa: E402
import llm  # noqa: E402
import store  # noqa: E402
import suggest  # noqa: E402
import textutil  # noqa: E402
import understand  # noqa: E402

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "NewsArchiveResearch/0.1 (local research project)"
CACHE = config.DATA / "people_cache.json"
MAX_PEOPLE = 4
SOURCE_CHARS = 2500
# words that make a page "sound like" a sports person; the article's own sport words count too
SPORTY = re.compile(r"athlet|player|olympi|champion|sport|coach|captain|medal|compet|represent|team|league|tournament"
                    r"|referee|umpire|official|president of|federation|association|grandmaster|racer|driver|jockey", re.I)

STR = {"type": "string"}
PROFILE_SCHEMA = understand.obj({
    "is_same_person": {"type": "string", "enum": ["yes", "no"]},        # an unknown value becomes "no": reject
    "sport": STR, "background": STR,
    "records": {"type": "array", "items": STR, "maxItems": 5},
    "current_status": STR})


# ---------------- Wikipedia ----------------
def wiki(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def title_matches(name: str, title: str) -> bool:
    """The page title is this person's name: all words of one are in the other (a middle name may differ)."""
    a, b = set(textutil.norm(name).split()), set(textutil.norm(title).split())
    return len(a) >= 2 and len(b) >= 2 and (a <= b or b <= a)


def sounds_sporty(description: str, text: str, sport: str) -> bool:
    head = f"{description} {text[:700]}"
    words = [w for w in textutil.norm(sport).split() if len(w) >= 4]
    return bool(SPORTY.search(head)) or any(w in textutil.norm(head) for w in words)


def lookup(name: str, sport: str = "") -> dict:
    """{"found": bool, "title", "url", "description", "text", "fetched", "reason"} for a person's Wikipedia page."""
    if len(textutil.norm(name).split()) < 2:
        return {"found": False, "reason": "only one name word is given, so no reliable page can be matched"}
    try:
        hits = wiki({"action": "query", "list": "search", "srsearch": f"{name} {sport}".strip(),
                     "srlimit": 5, "srprop": ""})["query"]["search"]
        for h in hits:
            if not title_matches(name, h["title"]):
                continue
            pages = wiki({"action": "query", "prop": "extracts|description|info", "inprop": "url", "exintro": 1,
                          "explaintext": 1, "redirects": 1, "titles": h["title"]})["query"]["pages"]
            p = pages[0] if pages else {}
            text, desc = (p.get("extract") or "").strip(), p.get("description") or ""
            if "may refer to" in text[:200] or not text:
                continue                                           # a disambiguation page
            if sounds_sporty(desc, text, sport):
                return {"found": True, "title": p["title"], "url": p.get("fullurl", ""), "description": desc,
                        "text": text[:SOURCE_CHARS], "fetched": date.today().isoformat(), "reason": ""}
        return {"found": False, "reason": "no Wikipedia page for this name that matches a sports person"}
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
        return {"found": False, "reason": f"lookup failed ({type(e).__name__}); check the internet connection",
                "failed": True}


def cached_lookup(name: str, sport: str, cache: dict) -> dict:
    key = textutil.norm(name)
    if key in cache:
        return cache[key]
    r = lookup(name, sport)
    if not r.get("failed"):                                        # a failed lookup is retried next time
        cache[key] = r
        store.write_json(CACHE, cache)
    return r


# ---------------- people of an article ----------------
def people_of(rec: dict) -> list[dict]:
    """The persons worth a card: named in full, in the article's order."""
    out, seen = [], set()
    for p in rec.get("facts", {}).get("participants", []):
        key = textutil.norm(p.get("name", ""))
        if p.get("kind") == "person" and len(key.split()) >= 2 and key not in seen:
            seen.add(key)
            out.append(p)
    return out[:MAX_PEOPLE]


def article_facts_about(rec: dict, name: str) -> list[dict]:
    """The figures the article gives for this person (the 'figures' step), matched by name."""
    n = set(textutil.norm(name).split())
    out = []
    for r in rec.get("facts", {}).get("results", []):
        w = set(textutil.norm(r.get("who", "")).split())
        if w and (w <= n or n <= w):
            out.append({k: r.get(k, "") for k in ("what", "value", "note")})
    return out


def build_profile(name: str, role: str, sport: str, page: dict, template: str) -> tuple[dict, dict | None]:
    """The model reads the page text only and writes the profile. Returns (profile, call log or None)."""
    base = {"found": False, "reason": page.get("reason", "")}
    if not page.get("found"):
        return base, None
    prompt = llm.fill(template, person_name=name, person_role=role or "not stated", person_sport=sport or "not stated",
                      source_text=page["text"])
    r = llm.call(prompt, PROFILE_SCHEMA, step="person_profile", max_new_tokens=450)
    log = {"person": name, "prompt_chars": len(prompt), "at": datetime.now().isoformat(timespec="seconds"), **r["log"]}
    if r["parsed"] is None:
        return {**base, "reason": "the model gave no usable answer"}, log
    p = extract.tidy(r["parsed"], PROFILE_SCHEMA)
    if p["is_same_person"] != "yes":
        return {**base, "reason": f'the page "{page["title"]}" is not clearly the person in the news'}, log
    text = " ".join([p["background"], p["current_status"], *p["records"]])
    bad = suggest.unmatched_numbers({"title": "", "detail": text}, page["text"])
    return {"found": True, "title": page["title"], "url": page["url"], "description": page["description"],
            "fetched": page["fetched"], "sport": p["sport"], "background": p["background"], "records": p["records"],
            "current_status": p["current_status"], "unmatched_numbers": bad, "verify": bool(bad),
            "source": "Wikipedia"}, log


def people_for_article(rec: dict, template: str, cache: dict) -> tuple[list[dict], list[dict]]:
    sport = rec.get("facts", {}).get("sport", "")
    people, logs = [], []
    for p in people_of(rec):
        page = cached_lookup(p["name"], sport, cache)
        profile, log = build_profile(p["name"], p.get("role", ""), sport, page, template)
        if log:
            logs.append({"article": rec["id"], **log})
        people.append({"name": p["name"], "role": p.get("role", ""), "affiliation": p.get("affiliation", ""),
                       "in_article": article_facts_about(rec, p["name"]), "profile": profile})
    return people, logs


def run(day: str, cls: str = "sports", article: str | None = None, limit: int | None = None,
        force: bool = False) -> None:
    template = (config.PROMPTS / cls / "11_person_profile.txt").read_text(encoding="utf-8")
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    data = store.load_day(cls, day)
    recs = [r for r in data["articles"].values() if "facts" in r and (not article or r["id"] == article)]
    todo = [r for r in recs if force or "people" not in r]
    if limit:
        todo = todo[:limit]
    print(f"{len(recs)} extracted {cls} articles on {day}; {len(todo)} to get people profiles with {llm.MODEL_NAME}")
    for n, rec in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {rec['id']}  {rec['headline'][:80]}")
        people, logs = people_for_article(rec, template, cache)
        rec["people"] = people
        store.save_day(data)
        store.log_calls(f"{day}_{cls}_people", logs)
        for p in people:
            pr = p["profile"]
            print(f'    {p["name"]:<26} ' + (f'profile from "{pr["title"]}"' + (" (verify numbers)" if pr["verify"] else "")
                                           if pr["found"] else f'no profile: {pr["reason"]}'))
        if not people:
            print("    no named persons")
    print(f"\nSaved {store.day_path(cls, day)}")


def main():
    config.utf8_console()
    ap = argparse.ArgumentParser(description="Step 4c: who the people in the news are")
    ap.add_argument("--date", help="the paper's date, YYYY-MM-DD")
    ap.add_argument("--class", dest="cls", default="sports")
    ap.add_argument("--article")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lookup", help="only test the Wikipedia lookup for this name")
    ap.add_argument("--sport", default="")
    a = ap.parse_args()
    if a.lookup:
        r = lookup(a.lookup, a.sport)
        print(json.dumps({k: (v[:300] if isinstance(v, str) else v) for k, v in r.items()}, indent=1, ensure_ascii=False))
        return
    if not a.date:
        ap.error("--date is required (or use --lookup NAME)")
    run(a.date, a.cls, a.article, a.limit, a.force)


if __name__ == "__main__":
    main()
