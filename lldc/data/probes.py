from __future__ import annotations
import json, random, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any

SPLIT_PATS = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ProbeSpec:
    out_path: Path
    n: int = 500
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
    seed: int = 17


def _mask_capital_spans(sent: str) -> Dict[str, str] | None:
    tokens = sent.split()
    best = (-1, -1)
    i = 0
    while i < len(tokens):
        if re.match(r"^[A-Z][A-Za-z\-]+", tokens[i]):
            j = i
            while j < len(tokens) and re.match(r"^[A-Z][A-Za-z\-]+", tokens[j]):
                j += 1
            if (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    if best == (-1, -1):
        return None
    i, j = best
    ans = " ".join(tokens[i:j])
    q = " ".join(tokens[:i] + ["[MASK]"] + tokens[j:])
    return {"q": q, "a": ans}


def generate_factual_probes(spec: ProbeSpec) -> Path:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(f"HuggingFace datasets not available: {e}")
    rnd = random.Random(spec.seed)
    ds = load_dataset(spec.dataset_name, spec.dataset_config)
    pool: List[Dict[str, str]] = []
    for row in ds["train"]:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        for sent in SPLIT_PATS.split(text):
            sent = sent.strip()
            if 20 <= len(sent) <= 240:
                m = _mask_capital_spans(sent)
                if m and 2 <= len(m["a"].split()) <= 4:
                    pool.append(m)
    rnd.shuffle(pool)
    spec.out_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.out_path.open("w", encoding="utf-8") as f:
        for item in pool[: spec.n]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return spec.out_path
