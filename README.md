# News — newspaper PDF → stored facts and meaning → questions by day

Takes a newspaper's e-paper PDF (first: *The Statesman*, Delhi edition), cuts it into articles, and runs **several
small prompts per class** on each article with a small local model (`Qwen3.5-0.8B`, CPU is enough). It stores the
facts and the meaning of every article per class and per day, so you can later ask about the news of a given day.

The first class is **Sports**. A class is ready once `prompts/<class>/extract_steps.json` exists. Adding Business
or World means adding a new prompt folder, with no code changes.

```
PDF ─► 1 split_pdf ─► 2 extract (7 calls) ─► 3 checks + dates ─► 4 understand (1 call) ─► 5 index ─► ask
       (code)          (model)                (code)               (model)                  (embeddings)  (model)
```

## Setup (from `graphrag/`)

```powershell
.\venv\Scripts\Activate.ps1
pip install -r news/requirements.txt
python news/download_models.py          # embedding model for search, ~130 MB -> news/models/
# Qwen3.5-0.8B is already in graphrag/models/; if not: python news/download_models.py --llm
```

Put newspaper PDFs in `news/papers/`. The name gives paper, edition and date: `The Statesman_Delhi_20261111.pdf`.

## Run, one step at a time (recommended the first time)

```powershell
# 1 split: check the review file against the printed page before going on
python news/split_pdf.py "news/papers/The Statesman_Delhi_20261111.pdf"
#   -> news/data/articles/2026-11-11/statesman-delhi.json and .review.txt

# 2-3 facts: one article first, to see the time per article on this machine
python news/extract.py --date 2026-11-11 --limit 1
python news/extract.py --date 2026-11-11                  # the rest (already-done articles are skipped)

# 4 meaning
python news/understand.py --date 2026-11-11

# look at everything stored
python news/show.py --date 2026-11-11
python news/show.py --date 2026-11-11 --article statesman-delhi-20261111-p13-02

# 5 search index, then questions
python news/search_index.py --date 2026-11-11
python news/ask.py "Why was the third ODI in Hamilton not played?"
python news/ask.py "How many votes did PT Usha get?" --date 2026-11-10
python news/ask.py "Who is in India's Test squad?" --no-llm       # only show which articles it would use
```

Or all steps at once: `python news/pipeline.py "news/papers/The Statesman_Delhi_20261111.pdf" [--limit 1]`.

## The steps

| Step | File | Model? | What it does |
|---|---|---|---|
| 1 | `split_pdf.py` | no | Reads the PDF's text layer (no OCR). Finds each page's class from its printed label ("S P O RT S" → sports). Sorts text blocks into headline, dateline, body, drop cap, kicker or furniture by font size. Gives every body block to the nearest headline above that covers its width. Reads each article column by column. Skips the advert page (different page size). |
| 2 | `extract.py` + `prompts/sports/1-7_*.txt` | 7 calls | **identify** (the sport in the model's own words, story type, gist) → **sport_profile** (the model explains how the sport works, what kind of contest the article reports, and the counting words the article uses) → **participants** (people, teams, roles) → **event** (competition, discipline, category, stage, venue, dates as written) → **outcome** (status, one-sentence result, winner) → **figures** (every score, time, distance, placing or vote, copied exactly, with a note for records or qualification) → **changes_next** (squad changes, quotes, what's next). Later calls get the earlier answers. |
| 3 | `extract.py` | no | Drops empty items. Looks every name and value up in the article (misses go to `checks.not_in_text`, nothing is deleted). Turns each written date ("Tuesday", "November 13 and 15") into a real day counted from the dateline. |
| 4 | `understand.py` + `8_meaning.txt` | 1 call | Summary, why it matters, happened/expected, key points, tags, and the questions the article answers. Only checked facts are given to it. |
| 5 | `search_index.py`, `ask.py` + `9_answer.txt` | 1 call per question | Filters to the day asked about. Ranks articles by keyword + meaning. The model answers only from the top 3, citing them. Code checks the cited articles and the evidence phrase. |

Shared code: `config.py` (paths, page labels), `llm.py` (the model, adapted from the travel branch's
`llm_backend.py`), `textutil.py` (text cleaning, the in-article check, dates), `store.py` (the per-day files).

## Where things are stored (`news/data/`, not committed)

| Path | What |
|---|---|
| `articles/<date>/<paper>.json` | step 1: every article of the paper (all classes) |
| `articles/<date>/<paper>.review.txt` | step 1 in plain text, plus text no article took, to check by eye |
| `store/<class>/<date>.json` | steps 2-4: per article `facts`, `dates`, `checks`, `meaning`, `runs` |
| `index/<class>/<date>.json` | step 5: one search vector per article |
| `logs/*.jsonl` | every model call: raw reply, tokens, seconds, cut off or not; every question asked |

**Which day an article belongs to.** It is stored under the paper's date (11 November). When you ask about a
day, it also counts as being about its dateline date (10 November, when it was written) and every date it
mentions (a Test starting 19 November). So "what happened on 10 November" finds the 11 November paper.

## Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `NEWS_BACKEND` | `hf` | `hf` = transformers on this machine, `ollama` = a model served by Ollama |
| `NEWS_HF_MODEL` | `graphrag/models/Qwen3.5-0.8B` | another Hugging Face model |
| `NEWS_OLLAMA_MODEL` | `qwen3.5:4b` | for `NEWS_BACKEND=ollama` |
| `NEWS_MAX_NEW_TOKENS` | `1024` | upper limit per call (each step sets its own, lower) |
| `NEWS_MAX_ARTICLE_CHARS` | `6000` | longer articles are cut before going to the model |
| `NEWS_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | embedding model for search |

## What to look at when testing

1. **The split** (`.review.txt` vs the printed page 13). Check that the 8 Sports headlines appear, that each text
   starts and ends where the printed article does, and that "Persistent rain…" starts with its drop cap.
   Then check that "TEXT NO ARTICLE TOOK" holds only page furniture. If an article is cut or mixed,
   the constants at the top of `split_pdf.py` are the knobs.
2. **Time per article** (`extract.py --limit 1`): 7 calls on CPU. This decides whether a whole paper is practical.
   In the same output, check the sport name and the "how it works" line, since every later call relies on them.
3. **Facts** (`show.py`): the `CHECKS` section lists values the model wrote that are not in the article.
   Many misses in one step means that step's prompt needs work.
4. **Dates** (`show.py`, the `date` lines): "Tuesday" in a 10 November dateline should become 2026-11-10.
5. **Answers**: compare with the page. "The stored news does not say" is the right answer for anything not
   on it.

## No list of sports: the model explains the sport

Nothing in the prompts or the code names a sport, so any sport or game can come in.

- **Call 1** writes the sport's name in its own words (`sport` is free text, lower-cased by code).
- **Call 2** (`2_sport_profile.txt`) makes the model explain it:
  - `how_it_works`: who competes, what is counted or measured, how a winner is decided. This comes from the
    article first and the model's own knowledge second. It is **not** checked against the article and is only
    used as background for the later prompts. It is never given to the answer prompt.
  - `contest_type`: what this article reports. One of two teams, two players, a ranked field, or no contest
    (an election, a squad announcement, a transfer).
  - `result_words`: the counting words the article itself uses, checked against the text.
- **Calls 4–6** get that explanation in their prompt, so they read "3/42" or "45.32s" knowing what it is.
- **Hints:** `extract_steps.json` holds one hint line per kind of contest (not per sport) for calls 4 and 6.

Every text fact a call returns can be used by a later prompt as `<its_key>`. Adding a prompt therefore means
adding a step to `extract_steps.json` and a prompt file, with no code change.

For a sport the small model does not know, `how_it_works` may be thin or wrong. The facts are still copied from
the article and checked against it, so a weak explanation makes extraction less complete but should not add false facts.

## Adding another class

Copy `prompts/sports/` to `prompts/<class>/`. Rewrite the prompts and the schemas in `extract_steps.json` for that
class (the keys must not repeat across steps). List which fields to check and which hold dates. Then run with
`--class <class>`. Each class keeps its own store, so questions stay within one class.
