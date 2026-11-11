"""
Step 1 - cut a newspaper PDF into articles (plain code, no model).

The Statesman's e-paper PDF has a real text layer, so no OCR is needed. For every page:

  1. Class of the page: the label printed at the top ("S P O RT S", "B U S I N E S S" ...), see config.PAGE_LABELS.
     Page 1 is "front"; a page of another size (the advert page) is skipped. --page-class 13=sports overrides.
  2. Text blocks are sorted into kinds by font size and wording:
        headline   clearly bigger than the page's body text      dateline   "STATESMAN NEWS SERVICE New Delhi, 10 November"
        kicker     a short all-caps label such as "BRIEFLY"      drop cap   the big first letter of an article
        body       the rest                                       skipped    masthead, page number and date, footer
  3. Every body block belongs to the nearest headline ABOVE it whose width covers it (columns under a headline).
     A block that no headline covers joins the article whose columns sit right next to it.
  4. Inside an article, blocks are read column by column, left to right, top to bottom.

Output: data/articles/<paper date>/<paper>.json          every article, with page, class, headline, dateline, text
        data/articles/<paper date>/<paper>.review.txt    the same in plain text + the text no article took,
                                                         to compare with the printed page

Run (from graphrag/):
    python news/split_pdf.py "news/papers/The Statesman_Delhi_20261111.pdf"
    python news/split_pdf.py <pdf> --pages 13                     only page 13
    python news/split_pdf.py <pdf> --page-class 13=sports         set a page's class by hand
The file name gives paper, edition and date ("The Statesman_Delhi_20261111.pdf"); else use --paper/--edition/--date.
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import store  # noqa: E402
import textutil  # noqa: E402

HEADLINE_RATIO = 1.3      # a headline's font is at least this many times the page's body font
DROPCAP_RATIO = 2.5       # a block of 1-2 letters this much bigger than the body font is a drop cap
MASTHEAD_RATIO = 5.0      # bigger than this (and not a drop cap) is the page's masthead / section label
SMALL_RATIO = 0.75        # text smaller than this (photo credits, tiny print) is left out
MIN_ARTICLE_CHARS = 80    # an "article" with less body text than this is dropped (listed in the review file)

DATELINE_RE = re.compile(
    r"^(?P<agency>[A-Z][A-Z.&'’ -]*[A-Z.])?\s*(?P<place>[A-Z][a-z][A-Za-z .'’-]*?),\s*"
    r"(?P<when>\d{1,2}\s+[A-Z][a-z]+|[A-Z][a-z]+\s+\d{1,2})\.?$")
SKIP_RE = re.compile(r"^\d{1,2}\s+New Delhi\b|\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+\d{1,2}\s+"
                     r"[A-Z][a-z]+\s+\d{4}$|Postal Reg\.|Printed and published|Printed at|RNI REG|RNI No|"
                     r"^(For more updates\s+)?www\.thestatesman\.com$|^epaper\.", re.I)
INLINE_DATELINE_RE = re.compile(r"^(?P<place>[A-Z][A-Za-z .'’-]{2,25}):\s+")


# ---------------- reading the PDF ----------------
def span_join(spans: list[dict]) -> str:
    """Join the spans of one line; add a space where the PDF left a gap but no space character."""
    out = ""
    for s in spans:
        t = s["text"]
        if out and not out.endswith(" ") and not t.startswith(" "):
            gap = s["bbox"][0] - prev["bbox"][2]
            if gap > 0.15 * s["size"]:
                out += " "
        out += t
        prev = s
    return out


def page_blocks(page) -> list[dict]:
    """The text blocks of a page: bbox, text, the largest font size, the text set in that size."""
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        lines = [[s for s in ln["spans"] if s["text"].strip()] for ln in b["lines"]]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        spans = [s for ln in lines for s in ln]
        size = max(s["size"] for s in spans)
        text = "\n".join(span_join(ln) for ln in lines)
        big = " ".join(span_join([s for s in ln if s["size"] >= 0.8 * size]) for ln in lines)
        blocks.append({"bbox": [round(v, 1) for v in b["bbox"]], "size": round(size, 1),
                       "font": max(spans, key=lambda s: s["size"])["font"], "text": text,
                       "big_text": textutil.clean_text(big), "chars": len(text.strip()), "spans": spans})
    return blocks


def body_size(blocks: list[dict]) -> float:
    """The font size most of the page's text is set in."""
    sizes = Counter()
    for b in blocks:
        for s in b["spans"]:
            if 7 <= s["size"] <= 20:
                sizes[round(s["size"])] += len(s["text"])
    return float(sizes.most_common(1)[0][0]) if sizes else 12.0


def page_label(page, blocks: list[dict]) -> str:
    """The section label at the top of the page ("SPORTS"), or ""."""
    top = page.rect.height * 0.15
    for b in blocks:
        if b["bbox"][1] > top:
            continue
        candidates = [span_join([s]) for s in b["spans"]] + [b["text"].replace("\n", " ")]
        for c in candidates:
            c = re.sub(r"www\.\S+|For more updates", " ", c, flags=re.I)
            key = re.sub(r"\s+", "", c).upper()
            if key in config.PAGE_LABELS:
                return key
    return ""


# ---------------- sorting blocks ----------------
def kind_of(b: dict, body: float, page_h: float) -> str:
    text = b["text"].strip()
    one_line = re.sub(r"\s+", " ", text)
    if b["bbox"][1] > page_h * 0.97 or SKIP_RE.search(one_line):
        return "skip"
    if re.sub(r"\s+", "", one_line).upper() in config.PAGE_LABELS:
        return "skip"
    if b["chars"] <= 2 and b["size"] >= DROPCAP_RATIO * body:
        return "dropcap"
    if b["size"] >= MASTHEAD_RATIO * body:
        return "skip"
    if b["chars"] < 90 and DATELINE_RE.match(one_line):
        return "dateline"
    if b["size"] >= HEADLINE_RATIO * body and len(b["big_text"]) >= 8:
        return "headline"
    if b["chars"] < 30 and one_line.isupper() and b["size"] >= 0.9 * body:
        return "kicker"
    if b["size"] < SMALL_RATIO * body:
        return "skip"
    return "body"


def overlap(a: list, b: list) -> float:
    """How much two boxes overlap left-to-right, in points."""
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def merge_headlines(heads: list[dict]) -> list[dict]:
    """A headline set as two blocks (one per line) becomes one."""
    heads = sorted(heads, key=lambda h: (h["bbox"][1], h["bbox"][0]))
    out = []
    for h in heads:
        prev = next((p for p in reversed(out)
                     if abs(p["size"] - h["size"]) <= 0.15 * p["size"]
                     and overlap(p["bbox"], h["bbox"]) >= 0.5 * min(p["bbox"][2] - p["bbox"][0], h["bbox"][2] - h["bbox"][0])
                     and -2 <= h["bbox"][1] - p["bbox"][3] <= 0.8 * h["size"]), None)
        if prev:
            prev["big_text"] = f'{prev["big_text"]} {h["big_text"]}'
            prev["bbox"] = [min(prev["bbox"][0], h["bbox"][0]), prev["bbox"][1],
                            max(prev["bbox"][2], h["bbox"][2]), max(prev["bbox"][3], h["bbox"][3])]
        else:
            out.append(dict(h))
    return out


def owner(b: dict, heads: list[dict]) -> int | None:
    """Index of the headline a block belongs to: the nearest one above it that covers at least half its width."""
    width = b["bbox"][2] - b["bbox"][0]
    best = None
    for i, h in enumerate(heads):
        if h["bbox"][1] <= b["bbox"][1] + 2 and overlap(b["bbox"], h["bbox"]) >= 0.5 * width:
            if best is None or h["bbox"][1] > heads[best]["bbox"][1]:
                best = i
    return best


def neighbour(b: dict, groups: dict[int, list[dict]], max_gap: float = 40) -> int | None:
    """For a block no headline covers: the article whose columns end right next to it, at the same height."""
    best, best_gap = None, max_gap
    for i, blocks in groups.items():
        for o in blocks:
            if o["bbox"][1] > b["bbox"][3] or o["bbox"][3] < b["bbox"][1]:      # not at the same height
                continue
            gap = max(0.0, b["bbox"][0] - o["bbox"][2], o["bbox"][0] - b["bbox"][2])
            if gap < best_gap:
                best, best_gap = i, gap
    return best


def attach_dropcaps(blocks: list[dict]) -> list[dict]:
    """Glue each drop cap ("P") to the start of the text beside it ("ersistent rain ...") -> "Persistent rain"."""
    caps = [b for b in blocks if b["kind"] == "dropcap"]
    rest = [dict(b) for b in blocks if b["kind"] != "dropcap"]
    for c in caps:
        h = c["bbox"][3] - c["bbox"][1]
        beside = [b for b in rest if b["kind"] == "body" and b["bbox"][0] >= c["bbox"][0] - 2
                  and abs(b["bbox"][1] - c["bbox"][1]) <= h]
        if beside:
            b = min(beside, key=lambda b: (b["bbox"][0] - c["bbox"][0], b["bbox"][1]))
            b["text"] = c["text"].strip() + b["text"].lstrip()
        else:
            rest.append({**c, "kind": "body"})
    return rest


def reading_order(blocks: list[dict]) -> list[dict]:
    """Columns left to right, each top to bottom. Blocks whose left edges are close are one column (the lines
    wrapped around a drop cap start a little to the right of the column's edge)."""
    if not blocks:
        return []
    widths = [b["bbox"][2] - b["bbox"][0] for b in blocks]
    tol = 0.5 * statistics.median(widths)
    columns: list[list[dict]] = []
    for b in sorted(blocks, key=lambda b: b["bbox"][0]):
        if columns and b["bbox"][0] - columns[-1][0]["bbox"][0] <= tol:
            columns[-1].append(b)
        else:
            columns.append([b])
    return [b for col in columns for b in sorted(col, key=lambda b: (round(b["bbox"][1]), b["bbox"][0]))]


def article_text(blocks: list[dict]) -> str:
    return textutil.clean_text("\n".join(b["text"] for b in reading_order(attach_dropcaps(blocks))))


# ---------------- one page ----------------
def split_page(page, page_no: int, cls: str, paper: dict) -> tuple[list[dict], list[str]]:
    """The articles of one page, and the text no article took."""
    blocks = page_blocks(page)
    body = body_size(blocks)
    for b in blocks:
        b["kind"] = kind_of(b, body, page.rect.height)
    heads = merge_headlines([b for b in blocks if b["kind"] == "headline"])
    groups: dict[int, list[dict]] = {i: [] for i in range(len(heads))}
    kickers: dict[int, str] = {}
    leftover = []
    for b in blocks:
        if b["kind"] == "kicker":                          # "BRIEFLY" sits just above its headline
            below = [i for i, h in enumerate(heads) if 0 <= h["bbox"][1] - b["bbox"][3] <= 3 * h["size"]
                     and overlap(b["bbox"], h["bbox"]) > 0]
            if below:
                kickers[min(below, key=lambda i: heads[i]["bbox"][1])] = b["text"].strip()
                continue
        if b["kind"] in ("body", "dateline", "dropcap", "kicker"):
            i = owner(b, heads)
            if i is None:
                leftover.append(b)
            else:
                groups[i].append(b)
    unassigned = []
    for b in leftover:
        i = neighbour(b, groups) if b["kind"] in ("body", "dropcap") else None
        if i is None:
            unassigned.append(textutil.clean_text(b["text"]))
        else:
            groups[i].append(b)

    articles = []
    for i, h in sorted(enumerate(heads), key=lambda x: (x[1]["bbox"][1], x[1]["bbox"][0])):
        parts = groups[i]
        dl = next((b for b in sorted(parts, key=lambda b: b["bbox"][1]) if b["kind"] == "dateline"), None)
        body_blocks = [b for b in parts if b["kind"] in ("body", "dropcap", "kicker")]
        text = article_text(body_blocks)
        agency, place, dl_date = "", "", None
        if dl:
            m = DATELINE_RE.match(re.sub(r"\s+", " ", dl["text"].strip()))
            agency = (m.group("agency") or "").strip()
            place, dl_date = textutil.parse_dateline(f'{m.group("place")}, {m.group("when")}', paper["date"])
        else:
            m = INLINE_DATELINE_RE.match(text)             # "New Delhi: Prime Minister ..." in a news brief
            if m:
                place = m.group("place").strip()
        box = [min(b["bbox"][k] for b in parts + [h]) if k < 2 else max(b["bbox"][k] for b in parts + [h]) for k in range(4)]
        articles.append({
            "page": page_no, "class": cls, "headline": h["big_text"].rstrip(":").strip(), "kicker": kickers.get(i, ""),
            "agency": agency, "dateline_place": place, "dateline_date": dl_date.isoformat() if dl_date else "",
            "text": text, "chars": len(text), "headline_size": h["size"], "bbox": [round(v) for v in box],
        })
    return articles, unassigned


# ---------------- the whole paper ----------------
def paper_info(pdf: Path, doc, args) -> dict:
    """Paper, edition and date from the file name ("The Statesman_Delhi_20261111.pdf"), the options, or page 2."""
    m = re.match(r"^(?P<paper>.+?)_(?P<edition>[A-Za-z ]+)_(?P<date>\d{8})$", pdf.stem)
    paper = args.paper or (m.group("paper") if m else pdf.stem)
    edition = args.edition or (m.group("edition") if m else "")
    if args.date:
        day = date.fromisoformat(args.date)
    elif m:
        day = datetime.strptime(m.group("date"), "%Y%m%d").date()
    else:
        found = None
        for p in doc:
            found = re.search(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(\d{1,2} [A-Z][a-z]+ \d{4})",
                              p.get_text())
            if found:
                break
        if not found:
            sys.exit("No date in the file name or the pages: pass --date YYYY-MM-DD")
        day = datetime.strptime(found.group(2), "%d %B %Y").date()
    slug = re.sub(r"[^a-z0-9]+", "-", f"{re.sub(r'(?i)^the ', '', paper)} {edition}".lower()).strip("-")
    return {"paper": paper, "edition": edition, "date": day, "slug": slug}


def split(pdf: Path, pages: list[int] | None = None, page_class: dict[int, str] | None = None, args=None) -> Path:
    """Cut the paper into articles; returns the path of the JSON file written."""
    import fitz                                            # PyMuPDF
    args = args or argparse.Namespace(paper=None, edition=None, date=None)
    page_class = page_class or {}
    doc = fitz.open(pdf)
    paper = paper_info(pdf, doc, args)
    common_size = Counter((round(p.rect.width), round(p.rect.height)) for p in doc).most_common(1)[0][0]
    out_pages, articles, unassigned = [], [], []
    for n, page in enumerate(doc, 1):
        if pages and n not in pages:
            continue
        size = (round(page.rect.width), round(page.rect.height))
        blocks = page_blocks(page)
        label = page_label(page, blocks)
        # page 1 is always the front page, whatever small label sits at its top
        cls = page_class.get(n) or ("front" if n == 1 else config.PAGE_LABELS.get(label, "unknown"))
        if size != common_size and n not in page_class:
            out_pages.append({"page": n, "label": label, "class": "skipped (other page size, likely an advert)"})
            continue
        arts, left = split_page(page, n, cls, paper)
        dropped = [a for a in arts if a["chars"] < MIN_ARTICLE_CHARS]
        arts = [a for a in arts if a["chars"] >= MIN_ARTICLE_CHARS]
        out_pages.append({"page": n, "label": label, "class": cls, "articles": len(arts),
                          "dropped_headlines": [a["headline"] for a in dropped]})
        articles.extend(arts)
        unassigned.extend({"page": n, "text": t} for t in left if len(t) > 3)

    day = paper["date"].isoformat()
    per_page = Counter()
    for a in articles:
        per_page[a["page"]] += 1
        a["id"] = f'{paper["slug"]}-{paper["date"]:%Y%m%d}-p{a["page"]:02d}-{per_page[a["page"]]:02d}'
        a.update({"paper": paper["paper"], "edition": paper["edition"], "paper_date": day})
        # the day the story was written: its dateline, else the day before the paper
        a["base_date"] = a["dateline_date"] or (paper["date"] - timedelta(days=1)).isoformat()
    result = {"paper": paper["paper"], "edition": paper["edition"], "paper_date": day, "source_pdf": str(pdf),
              "made_at": datetime.now().isoformat(timespec="seconds"), "pages": out_pages,
              "articles": articles, "unassigned": unassigned}
    out = config.ARTICLES / day / f'{paper["slug"]}.json'
    store.write_json(out, result)
    out.with_suffix(".review.txt").write_text(review_text(result), encoding="utf-8")
    return out


def review_text(r: dict) -> str:
    """The articles as plain text, page by page, to compare with the printed paper."""
    lines = [f'{r["paper"]} ({r["edition"]}) {r["paper_date"]}', ""]
    for p in r["pages"]:
        lines.append(f'Page {p["page"]}: {p["class"]}' + (f' (label {p["label"]})' if p.get("label") else "")
                     + (f', {p["articles"]} articles' if "articles" in p else ""))
        for h in p.get("dropped_headlines", []):
            lines.append(f"    dropped (too little text under it): {h}")
    for a in r["articles"]:
        lines += ["", "=" * 100, f'{a["id"]}  page {a["page"]}  [{a["class"]}]  {a["chars"]} chars',
                  (f'[{a["kicker"]}] ' if a["kicker"] else "") + a["headline"],
                  f'{a["agency"] or "-"} | {a["dateline_place"] or "-"} | {a["dateline_date"] or "no date"}',
                  "-" * 100, a["text"]]
    if r["unassigned"]:
        lines += ["", "=" * 100, "TEXT NO ARTICLE TOOK (check it is only captions, adverts or page furniture)"]
        lines += [f'p{u["page"]}: {u["text"][:300]}' for u in r["unassigned"]]
    return "\n".join(lines) + "\n"


def main():
    config.utf8_console()
    ap = argparse.ArgumentParser(description="Step 1: cut a newspaper PDF into articles")
    ap.add_argument("pdf")
    ap.add_argument("--pages", help="only these pages, e.g. 13 or 2,3,13")
    ap.add_argument("--page-class", action="append", default=[], help="set a page's class, e.g. 13=sports (repeatable)")
    ap.add_argument("--paper"), ap.add_argument("--edition"), ap.add_argument("--date", help="YYYY-MM-DD")
    args = ap.parse_args()
    pages = [int(x) for x in args.pages.split(",")] if args.pages else None
    page_class = {int(k): v for k, v in (x.split("=", 1) for x in args.page_class)}
    out = split(Path(args.pdf), pages, page_class, args)
    r = json.loads(out.read_text(encoding="utf-8"))
    for p in r["pages"]:
        print(f'page {p["page"]:>2}  {p["class"]:<12} {p.get("articles", "")}')
    by_class = Counter(a["class"] for a in r["articles"])
    print(f'\n{len(r["articles"])} articles ({", ".join(f"{k} {v}" for k, v in by_class.most_common())}), '
          f'{len(r["unassigned"])} text blocks no article took')
    print(f"Saved {out}\nCheck {out.with_suffix('.review.txt')}")


if __name__ == "__main__":
    main()
