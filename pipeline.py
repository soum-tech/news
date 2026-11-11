"""
All steps for one paper and one class, in order:
    1 split_pdf.py      cut the PDF into articles                      (code)
    2-3 extract.py      seven calls per article -> checked facts -> store  (model + code)
    4 understand.py     one call per article -> meaning -> store            (model)
    5 search_index.py   search vectors for the day                         (embedding model)
Each step skips articles already done, so a stopped run can simply be started again.
Then ask questions with ask.py.

Run (from graphrag/):
    python news/pipeline.py "news/papers/The Statesman_Delhi_20261111.pdf"
    python news/pipeline.py <pdf> --limit 1          one article only (to see the time per article first)
    python news/pipeline.py <pdf> --pages 13         split only page 13
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main():
    config.utf8_console()
    ap = argparse.ArgumentParser(description="Steps 1-5 for one paper and one class")
    ap.add_argument("pdf")
    ap.add_argument("--class", dest="cls", default="sports")
    ap.add_argument("--pages", help="split only these pages, e.g. 13")
    ap.add_argument("--page-class", action="append", default=[], help="e.g. 13=sports")
    ap.add_argument("--limit", type=int, help="at most this many articles in steps 2-4")
    ap.add_argument("--force", action="store_true", help="redo steps 2-4 for articles already stored")
    a = ap.parse_args()

    import extract
    import search_index
    import split_pdf
    import understand

    t0 = time.time()
    print("== Step 1: split the paper")
    pages = [int(x) for x in a.pages.split(",")] if a.pages else None
    page_class = {int(k): v for k, v in (x.split("=", 1) for x in a.page_class)}
    out = split_pdf.split(Path(a.pdf), pages, page_class)
    r = json.loads(out.read_text(encoding="utf-8"))
    day = r["paper_date"]
    n = sum(1 for x in r["articles"] if x["class"] == a.cls)
    print(f"{len(r['articles'])} articles, {n} of them {a.cls}. Check {out.with_suffix('.review.txt')}")
    if not n:
        sys.exit(f"No {a.cls} articles found. Look at the page classes in the review file; set one with --page-class.")

    print(f"\n== Steps 2-3: facts ({a.cls}, {day})")
    extract.run(day, a.cls, paper=None, limit=a.limit, force=a.force)
    print(f"\n== Step 4: meaning")
    understand.run(day, a.cls, limit=a.limit, force=a.force)
    print(f"\n== Step 5: search index")
    search_index.build(day, a.cls)
    print(f"\nDone in {time.time() - t0:.0f}s. Look:  python news/show.py --date {day}")
    print(f'Ask:   python news/ask.py "your question" --class {a.cls}')


if __name__ == "__main__":
    main()
