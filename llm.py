"""
The small local model used by every step. Adapted from the traveler brief (branch somtrav,
llm_calls/traveler_brief/llm_backend.py). Two backends, chosen with NEWS_BACKEND:

    hf      (default)  Hugging Face transformers. Loads graphrag/models/Qwen3.5-0.8B if it is there (download it
                       with `python model.py` from graphrag/), else Qwen/Qwen3.5-0.8B from the Hub. GPU if there is
                       one, else CPU. NEWS_HF_MODEL picks another model.
    ollama             a model served by Ollama (NEWS_OLLAMA_MODEL, default qwen3.5:4b); JSON Schema enforced by Ollama.

transformers cannot enforce a JSON Schema, so for "hf" the expected JSON shape is shown to the model in a system
message, the reply is parsed leniently (json_repair) and then forced into the schema by conform(): missing keys get
empty values, extra keys are dropped, an unknown enum value becomes the enum's last value, lists are cut to maxItems.

Other settings: NEWS_MAX_NEW_TOKENS (default 1024, a step can ask for less).
"""

import json
import os
import re
import time

import config

BACKEND = os.getenv("NEWS_BACKEND", "hf").lower()
_LOCAL = config.REPO / "models" / "Qwen3.5-0.8B"
HF_MODEL = os.getenv("NEWS_HF_MODEL") or (str(_LOCAL) if (_LOCAL / "config.json").exists() else "Qwen/Qwen3.5-0.8B")
OLLAMA_MODEL = os.getenv("NEWS_OLLAMA_MODEL", "qwen3.5:4b")
MODEL_NAME = HF_MODEL if BACKEND == "hf" else OLLAMA_MODEL
MAX_NEW_TOKENS = int(os.getenv("NEWS_MAX_NEW_TOKENS", "1024"))


# ---------------- JSON shape helpers ----------------
def skeleton(schema: dict):
    """An example of the JSON shape: {"sport": "cricket|football|...", "participants": [{"name": ""}]}."""
    t = schema.get("type")
    if t == "object":
        return {k: skeleton(s) for k, s in schema.get("properties", {}).items()}
    if t == "array":
        return [skeleton(schema["items"])]
    if t == "integer":
        return 0
    if "enum" in schema:
        return "|".join(schema["enum"])
    return ""


def _find_key(v, key: str):
    """The first value stored under `key` anywhere inside v (depth-first), or None."""
    if isinstance(v, dict):
        if key in v:
            return v[key]
        v = list(v.values())
    if isinstance(v, list):
        for x in v:
            found = _find_key(x, key)
            if found is not None:
                return found
    return None


def conform(v, schema: dict, root: bool = True):
    """Force a parsed reply into the schema's shape (see the module docstring). At the top level, a key the model
    put in the wrong place (nested one level too deep) is found and moved back up."""
    t = schema.get("type")
    if t == "object":
        props = schema.get("properties", {})
        if isinstance(v, list) and len(props) == 1:          # {"quotes": [...]} answered as just [...]
            v = {next(iter(props)): v}
        v = v if isinstance(v, dict) else {}
        return {k: conform(v.get(k) if k in v or not root else _find_key(v, k), s, False) for k, s in props.items()}
    if t == "array":
        items = v if isinstance(v, list) else ([] if v in (None, "", {}) else [v])
        return [conform(x, schema["items"], False) for x in items][: schema.get("maxItems", 1000)]
    if t == "integer":
        try:
            return int(float(str(v).strip()))
        except (TypeError, ValueError):
            return 0
    if isinstance(v, list):
        v = ", ".join(str(x) for x in v if x not in (None, ""))
    elif isinstance(v, dict):
        v = json.dumps(v, ensure_ascii=False) if v else ""
    v = "" if v is None else str(v).strip()
    if "enum" in schema:
        low = {e.lower(): e for e in schema["enum"]}
        return low.get(v.lower().replace(" ", "_").replace("-", "_"), low.get(v.lower(), schema["enum"][-1]))
    return v


def parse_json(text: str):
    """The first JSON value in a model reply: think blocks and ``` fences removed, broken JSON repaired."""
    text = re.sub(r"(?s)<think>.*?</think>", "", text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    try:
        import json_repair
        v = json_repair.loads(text[start:])
        return v if v not in ("", None) else None
    except ImportError:
        try:
            return json.loads(text[start: text.rfind("}") + 1])
        except json.JSONDecodeError:
            return None


# ---------------- Hugging Face transformers ----------------
_hf = {}


def load_hf():
    """Load the model and processor once (the first call takes a while)."""
    if _hf:
        return _hf
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    t0 = time.time()
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "cuda"
    else:
        dtype, device_map = torch.float32, "cpu"
    _hf["processor"] = AutoProcessor.from_pretrained(HF_MODEL)
    _hf["model"] = AutoModelForImageTextToText.from_pretrained(HF_MODEL, dtype=dtype, device_map=device_map).eval()
    _hf["device"] = device_map
    print(f"Loaded {HF_MODEL} on {device_map} ({dtype}) in {time.time() - t0:.0f}s")
    return _hf


def _loop_stopper(n_in: int, period_max: int = 24, repeats: int = 6):
    """A StoppingCriteria that ends generation once the last tokens just repeat one short pattern
    (small models get stuck writing '"tags":[],' forever); json_repair then closes the JSON."""
    import torch
    from transformers import StoppingCriteria

    class LoopStop(StoppingCriteria):
        looped = False

        def __call__(self, input_ids, scores, **kwargs):
            ids = input_ids[0, n_in:].tolist()
            span = period_max * repeats
            if len(ids) >= span and len(ids) % 8 == 0:
                seg = ids[-span:]
                LoopStop.looped = any(all(seg[i] == seg[i - p] for i in range(p, span)) for p in range(1, period_max + 1))
            return torch.full((input_ids.shape[0],), LoopStop.looped, dtype=torch.bool, device=input_ids.device)
    return LoopStop()


def hf_chat(prompt: str, schema: dict | None, max_new_tokens: int) -> dict:
    import torch
    from transformers import StoppingCriteriaList
    m = load_hf()
    messages = []
    if schema:
        messages.append({"role": "system", "content": [{"type": "text", "text":
                         "Reply with ONE JSON object only, no other text, in exactly this shape "
                         "(\"a|b\" means one of the listed values; lists may have any number of items, or none):\n"
                         + json.dumps(skeleton(schema), ensure_ascii=False)}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    inputs = m["processor"].apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                return_tensors="pt", enable_thinking=False).to(m["device"])
    n_in = inputs["input_ids"].shape[1]
    stop = _loop_stopper(n_in)
    with torch.inference_mode():
        out = m["model"].generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05,
                                  stopping_criteria=StoppingCriteriaList([stop]))
    gen = out[0, n_in:]
    text = m["processor"].decode(gen, skip_special_tokens=True)
    return {"raw": text, "input_tokens": n_in, "output_tokens": len(gen), "looped": type(stop).looped,
            "truncated": len(gen) >= max_new_tokens or type(stop).looped}


# ---------------- Ollama ----------------
def ollama_chat(prompt: str, schema: dict | None, max_new_tokens: int) -> dict:
    import ollama
    r = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}], format=schema, think=False,
                    keep_alive="30m", options={"temperature": 0, "num_ctx": 16384, "num_predict": max_new_tokens})
    return {"raw": r.message.content or "", "input_tokens": r.prompt_eval_count, "output_tokens": r.eval_count,
            "looped": False, "truncated": r.done_reason == "length"}


# ---------------- One call, logged ----------------
def call(prompt: str, schema: dict | None = None, step: str = "", max_new_tokens: int | None = None) -> dict:
    """One model call -> {"parsed" (dict in the schema's shape, or None), "raw", "log"}.
    "log" holds what is worth keeping: step, model, tokens, seconds, whether it was cut off, the raw reply."""
    t0 = time.time()
    n = min(max_new_tokens or MAX_NEW_TOKENS, MAX_NEW_TOKENS)
    r = (hf_chat if BACKEND == "hf" else ollama_chat)(prompt, schema, n)
    parsed = None
    if schema:
        v = parse_json(r["raw"])
        parsed = conform(v, schema) if isinstance(v, (dict, list)) else None
    log = {"step": step, "model": MODEL_NAME, "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
           "seconds": round(time.time() - t0, 1), "truncated": r["truncated"], "looped": r.get("looped", False),
           "ok": schema is None or parsed is not None, "raw": r["raw"][:4000]}
    return {"parsed": parsed, "raw": r["raw"], "log": log}


def fill(template: str, **values) -> str:
    """Put values into a prompt: <headline> -> values["headline"]. One pass over the template, so a "<word>"
    that happens to be inside an inserted value (an article, a model's earlier answer) is left alone."""
    return re.sub(r"<(\w+)>", lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), template)
