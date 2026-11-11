"""
Step 5b - ask a question about the stored news of one class.

    1. Day: --date, or the day named in the question ("on 10 November", "yesterday"). With a day, only articles
       about that day are searched: the paper's date, the dateline date, or any date the article mentions
       (store.article_days). Without one, every stored day is searched.
    2. Rank: keyword match (names, numbers, rare words count most) + meaning match (the search vectors of
       search_index.py, if built). The best --top articles (default 3) go on.
    3. Answer (model, prompts/<class>/9_answer.txt): only from those articles. Code then checks that the articles
       it cites were really given to it and that its evidence phrase is in the article.

Run (from graphrag/):
    python news/ask.py "Why was the third ODI in Hamilton not played?"
    python news/ask.py "How many votes did PT Usha get?" --date 2026-11-10
    python news/ask.py "Who replaced Savita in the squad?" --no-llm        only show which articles it would use
Every question and answer is logged in data/logs/ask.jsonl.
"""

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import store  # noqa: E402
import textutil  # noqa: E402

ARTICLE_CHARS = 2500       # each article's text in the answer prompt
STOP = set("""a an the of in on at to for from by with and or but is are was were be been being has have had do does
did what which who whom whose when where why how many much did not no yes it its this that these those there their
they them he she his her him i me my we our you your about after before over under into than then so as if any
some all can could would should will shall may might news article articles paper day today yesterday tell please
happened happen""".split())


def tokens(s: str) -> list[str]:
    return [w for w in textutil.norm(s).split() if w not in STOP and (len(w) > 1 or w.isdigit())]


def doc_text(rec: dict) -> str:
    f, m = rec.get("facts", {}), rec.get("meaning") or {}
    return " ".join([rec["headline"], rec["headline"], rec.get("text", ""), json.dumps(f, ensure_ascii=False),
                     json.dumps(m, ensure_ascii=False)])


def keyword_scores(question: str, recs: list[dict]) -> list[float]:
    """Share of the question's words found in each article, each word weighted by how rare it is (idf)."""
    q = set(tokens(question))
    docs = [set(tokens(doc_text(r))) for r in recs]
    if not q or not docs:
        return [0.0] * len(recs)
    n = len(docs)
    idf = {w: math.log(1 + n / (1 + sum(w in d for d in docs))) for w in q}
    total = sum(idf.values()) or 1.0
    return [sum(idf[w] for w in q if w in d) / total for d in docs]


def semantic_scores(question: str, recs: list[dict], cls: str) -> list[float] | None:
    """Cosine similarity between the question and each article's search vector; None without an index."""
    import search_index
    vectors = {}
    for day in {r["paper_date"] for r in recs}:
        p = search_index.index_path(cls, day)
        if p.exists():
            vectors.update(json.loads(p.read_text(encoding="utf-8"))["vectors"])
    if not vectors:
        print("(no search index yet: run search_index.py; using keyword search only)")
        return None
    q = search_index.embed([question], query=True)
    if q is None:
        return None
    q = [float(x) for x in q[0]]
    return [sum(a * b for a, b in zip(q, vectors[r["id"]])) if r["id"] in vectors else 0.0 for r in recs]


def rank(question: str, recs: list[dict], cls: str) -> list[tuple[float, dict]]:
    kw = keyword_scores(question, recs)
    sem = semantic_scores(question, recs, cls)
    scores = [0.5 * k + 0.5 * s for k, s in zip(kw, sem)] if sem else kw
    return sorted(zip(scores, recs), key=lambda x: -x[0])


def articles_block(top: list[dict]) -> str:
    parts = []
    for i, r in enumerate(top, 1):
        m, f = r.get("meaning") or {}, r.get("facts", {})
        text = r.get("text", "")
        if len(text) > ARTICLE_CHARS:
            text = text[:ARTICLE_CHARS] + " [...]"
        results = "; ".join(f'{x.get("who", "")} {x.get("what", "")}: {x.get("value", "")}' for x in f.get("results", []))
        parts.append("\n".join(x for x in [
            f'[A{i}] {r["headline"]}',
            f'Paper: {r["paper"]} ({r["edition"]}), {r["paper_date"]}, page {r["page"]}. '
            f'Dateline: {r.get("dateline_place") or "-"}, {r.get("dateline_date") or "-"}',
            f'Sport: {f["sport"]}' if f.get("sport") else "",   # not how_it_works: that is the model's own knowledge
            f'Summary: {m["summary"]}' if m.get("summary") else "",
            f'Winner: {f["winner"]}' if f.get("winner") else "",
            f"Results: {results}" if results else "",
            f"Text: {text}"] if x))
    return "\n\n".join(parts)


ANSWER_SCHEMA = {"type": "object", "properties": {
    "found": {"type": "string", "enum": ["yes", "no"]},
    "answer": {"type": "string"},
    "articles": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    "evidence": {"type": "string"}}, "required": ["found", "answer", "articles", "evidence"]}


def answer(question: str, asked_day: str, top: list[dict], cls: str) -> dict:
    import llm
    template = (config.PROMPTS / cls / "9_answer.txt").read_text(encoding="utf-8")
    prompt = llm.fill(template, question=question, asked_day=asked_day, articles=articles_block(top))
    r = llm.call(prompt, ANSWER_SCHEMA, step="answer", max_new_tokens=400)
    a = r["parsed"] or {"found": "no", "answer": "(the model gave no usable answer)", "articles": [], "evidence": ""}
    labels = {f"A{i}": rec for i, rec in enumerate(top, 1)}
    cited = [x.strip("[] ") for x in a["articles"]]
    a["sources"] = [labels[x] for x in cited if x in labels]
    a["unknown_labels"] = [x for x in cited if x not in labels]
    texts = " ".join(f'{s["headline"]} {s.get("text", "")}' for s in (a["sources"] or top))
    a["evidence_in_article"] = bool(a["evidence"]) and textutil.found_in(a["evidence"], textutil.norm(texts))
    a["log"] = r["log"]
    return a


def main():
    config.utf8_console()
    ap = argparse.ArgumentParser(description="Step 5b: ask a question about the stored news")
    ap.add_argument("question")
    ap.add_argument("--class", dest="cls", default="sports")
    ap.add_argument("--date", help="the day asked about, YYYY-MM-DD (else taken from the question, if it names one)")
    ap.add_argument("--top", type=int, default=3, help="articles given to the model")
    ap.add_argument("--no-llm", action="store_true", help="only show the articles that would be used")
    a = ap.parse_args()

    recs = [r for r in store.all_articles(a.cls) if r.get("facts")]
    if not recs:
        sys.exit(f"Nothing extracted for {a.cls} yet: run extract.py first")
    day = date.fromisoformat(a.date) if a.date else textutil.find_date(a.question, date.today())
    if day:
        recs = [r for r in recs if day.isoformat() in store.article_days(r)]
        print(f"Day asked about: {day} ({len(recs)} stored {a.cls} articles are about that day)")
        if not recs:
            print(f"No stored {a.cls} news is about {day}. Stored paper dates: {', '.join(store.days(a.cls))}")
            return
    else:
        print(f"No day named: searching all {len(recs)} stored {a.cls} articles")

    ranked = rank(a.question, recs, a.cls)
    top = [r for s, r in ranked[: a.top] if s > 0] or [r for _, r in ranked[: a.top]]
    print("\nBest matches:")
    for s, r in ranked[: max(a.top, 5)]:
        print(f'  {s:5.2f}  {r["paper_date"]} p{r["page"]}  {r["headline"][:90]}')
    if a.no_llm:
        return

    res = answer(a.question, day.isoformat() if day else "not given", top, a.cls)
    print(f'\nAnswer ({"found" if res["found"] == "yes" else "not found"}): {res["answer"]}')
    for s in res["sources"]:
        print(f'  source: {s["paper"]} ({s["edition"]}) {s["paper_date"]}, page {s["page"]}: "{s["headline"]}"')
    if res["evidence"]:
        print(f'  evidence: "{res["evidence"]}" '
              + ("(found in the article)" if res["evidence_in_article"] else "(NOT found word for word in the article)"))
    if res["unknown_labels"]:
        print(f'  warning: the model cited articles it was not given: {res["unknown_labels"]}')
    store.log_calls("ask", [{"at": datetime.now().isoformat(timespec="seconds"), "question": a.question,
                             "day": day.isoformat() if day else "", "class": a.cls,
                             "given": [r["id"] for r in top], "answer": res["answer"], "found": res["found"],
                             "sources": [s["id"] for s in res["sources"]], "evidence": res["evidence"],
                             "evidence_in_article": res["evidence_in_article"], "seconds": res["log"]["seconds"],
                             "raw": res["log"]["raw"]}])


if __name__ == "__main__":
    main()
