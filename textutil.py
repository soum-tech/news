"""
Text and date helpers shared by the steps (plain code, no model).

    clean_text()       tidy text taken out of the PDF (spaced-out letters, soft hyphens, line breaks)
    norm(), found_in() is a value really written in the article? (used to flag what the model made up)
    parse_dateline()   "New Delhi, 10 November" -> place + date (year taken from the paper's date)
    resolve_date()     "Tuesday", "November 13 and 15", "yesterday" -> ISO dates, from a base date
    find_date()        the date asked about in a question ("on 10 November", "yesterday")
"""

import re
from datetime import date, timedelta

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s*(\d{4}))?"
DAY_MONTH = re.compile(rf"\b{_DAY}\s+{_MON}{_YEAR}\b", re.I)          # 13 November (2026)
MONTH_DAY = re.compile(rf"\b{_MON}\s+{_DAY}\b{_YEAR}", re.I)          # November 13 (, 2026)
ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
NUMERIC = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})\b")      # 19/11/2026, day first
WEEKDAY = re.compile(r"\b(?:(last|next|this|coming|previous)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)


# ---------------- text ----------------
def clean_text(s: str) -> str:
    """Tidy text from the PDF: soft hyphens out, words broken over lines joined, spaces collapsed, and words set
    with spaced-out letters ("s e p a r a t i s t", from justified columns) put back together."""
    s = s.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = re.sub(r"(?<=[a-z])-\s*\n\s*(?=[a-z])", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(?:[A-Za-z] ){3,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), s)
    return s.strip()


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def found_in(value: str, text_norm: str) -> bool:
    """Is the value written in the text? Exact (after normalising), or every word of it present.
    text_norm must already be norm(text)."""
    v = norm(value)
    if not v or v in text_norm:
        return True
    words = [w for w in v.split() if len(w) > 1]
    return bool(words) and all(re.search(rf"\b{re.escape(w)}\b", text_norm) for w in words)


# ---------------- dates ----------------
def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _fit_year(d: date, base: date, prefer: str, year_given: bool) -> date:
    """A date written without a year gets the base date's year, moved by one year when it would otherwise be far
    on the wrong side ("5 January" in a December paper means next January)."""
    if year_given:
        return d
    if d - base > timedelta(days=180):
        return _safe_date(d.year - 1, d.month, d.day) or d
    if base - d > timedelta(days=180) or (prefer == "future" and base - d > timedelta(days=60)):
        return _safe_date(d.year + 1, d.month, d.day) or d
    return d


def absolute_dates(text: str, base: date, prefer: str = "past") -> list[date]:
    """Every calendar date written in the text, in order. "November 13 and 15" and "19 to 28 November" give both
    days."""
    found = []                                   # (position, date)
    for rx, day_first in ((DAY_MONTH, True), (MONTH_DAY, False)):
        for m in rx.finditer(text):
            day, mon = (m.group(1), m.group(2)) if day_first else (m.group(2), m.group(1))
            year = m.group(3)
            d = _safe_date(int(year) if year else base.year, MONTHS[mon.lower().rstrip(".")], int(day))
            if d:
                found.append((m.start(), _fit_year(d, base, prefer, bool(year))))
                # a second day right before or after it in the same month: "13 and 15 November", "November 13 and 15"
                rest = text[m.end(): m.end() + 14] if not day_first else ""
                before = text[max(0, m.start() - 14): m.start()] if day_first else ""
                for extra in (re.match(r"\s*(?:and|to|till|until|-|–|&)\s*(\d{1,2})(?:st|nd|rd|th)?\b", rest),
                              re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:and|to|till|until|-|–|&)\s*$", before)):
                    if extra:
                        d2 = _safe_date(d.year, d.month, int(extra.group(1)))
                        if d2:
                            found.append((m.start() + (1 if rest else -1), _fit_year(d2, base, prefer, bool(year))))
    for m in ISO.finditer(text):
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            found.append((m.start(), d))
    for m in NUMERIC.finditer(text):
        y = int(m.group(3))
        d = _safe_date(y + 2000 if y < 100 else y, int(m.group(2)), int(m.group(1)))
        if d:
            found.append((m.start(), d))
    out = []
    for _, d in sorted(found, key=lambda x: x[0]):
        if d not in out:
            out.append(d)
    return out


def resolve_date(text: str, base: date, prefer: str = "past") -> dict:
    """A date as written in an article ("Tuesday", "November 13 and 15", "last Saturday", "yesterday") -> ISO.
    base is the day the article was written (its dateline). prefer is "past" for things that happened and "future"
    for things still to come; it decides which Tuesday a bare "Tuesday" means.
    Returns {"date": "YYYY-MM-DD" or "", "end": last day of a range or "", "how": absolute|relative|weekday|unresolved}."""
    t = (text or "").strip()
    low = t.lower()
    if not t:
        return {"date": "", "end": "", "how": "unresolved"}
    days = absolute_dates(t, base, prefer)
    if days:
        return {"date": days[0].isoformat(), "end": days[-1].isoformat() if len(days) > 1 else "", "how": "absolute"}
    for words, delta in ((("day after tomorrow",), 2), (("day before yesterday",), -2), (("tomorrow",), 1),
                         (("yesterday",), -1), (("today", "tonight", "this evening", "this morning"), 0)):
        if any(w in low for w in words):
            return {"date": (base + timedelta(days=delta)).isoformat(), "end": "", "how": "relative"}
    m = WEEKDAY.search(t)
    if m:
        word, wd = (m.group(1) or "").lower(), WEEKDAYS.index(m.group(2).lower())
        if word in ("next", "coming") or (prefer == "future" and word not in ("last", "previous")):
            ahead = (wd - base.weekday()) % 7
            if word in ("next", "coming") and ahead == 0:
                ahead = 7
            d = base + timedelta(days=ahead)
        else:
            back = (base.weekday() - wd) % 7
            if word in ("last", "previous") and back == 0:
                back = 7
            d = base - timedelta(days=back)
        return {"date": d.isoformat(), "end": "", "how": "weekday"}
    return {"date": "", "end": "", "how": "unresolved"}


def parse_dateline(text: str, paper_date: date) -> tuple[str, date | None]:
    """"New Delhi, 10 November" -> ("New Delhi", 2026-11-10). The year is the paper's (one less if that would put
    the dateline after the paper)."""
    m = re.match(rf"^\s*(.*?),?\s*{_DAY}\s+{_MON}\s*$", text.strip(), re.I) or \
        re.match(rf"^\s*(.*?),?\s*{_MON}\s+{_DAY}\s*$", text.strip(), re.I)
    if not m:
        return text.strip().rstrip(","), None
    place = m.group(1).strip().rstrip(",")
    a, b = m.group(2), m.group(3)
    day, mon = (a, b) if a.isdigit() else (b, a)
    d = _safe_date(paper_date.year, MONTHS[mon.lower().rstrip(".")], int(day))
    if d and d > paper_date + timedelta(days=1):
        d = _safe_date(paper_date.year - 1, d.month, d.day)
    return place, d


def find_date(question: str, today: date) -> date | None:
    """The day a question asks about: "on 10 November", "2026-11-10", "yesterday", "last Tuesday" (all in the past,
    relative to today). None if the question names no day."""
    r = resolve_date(question, today, prefer="past")
    if r["how"] == "unresolved":
        return None
    if r["how"] == "weekday" and not re.search(r"\b(last|previous|this|on)\s+(mon|tues|wednes|thurs|fri|satur|sun)day\b",
                                               question, re.I):
        return None                                # a bare weekday inside a question is too vague to filter on
    return date.fromisoformat(r["date"])
