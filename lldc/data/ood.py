# lldc/data/ood.py

from __future__ import annotations
from pathlib import Path
import random
from datasets import load_dataset


def synth_math(n: int, rnd: random.Random) -> list[str]:
    return [
        f"Solve: {rnd.randint(10,999)} + {rnd.randint(10,999)} = ? Explain each step in words."
        for _ in range(n)
    ]


def synth_code(n: int, rnd: random.Random) -> list[str]:
    return [
        f"def f_{i}(x):\n    # compute square then add three\n    return x*x + 3\n"
        for i in range(n)
    ]


def from_stack(n: int) -> list[str]:
    try:
        ds = load_dataset(
            "bigcode/the-stack", data_dir="data/python", split="train[:2000]"
        )
        out = []
        for r in ds:
            t = (r.get("content") or "").strip()
            if t:
                out.append(t)
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def from_math(n: int) -> list[str]:
    try:
        ds = load_dataset("math_dataset", "all", split="train[:5000]")
        out = []
        for r in ds:
            q = (r.get("question") or "").strip()
            if q:
                out.append(q)
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def ensure_ood(
    out: Path, n: int = 500, mode: str = "synthetic", seed: int = 17
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)
    if mode == "stack":
        lines = from_stack(n) or synth_code(n, rnd)
    elif mode == "math":
        lines = from_math(n) or synth_math(n, rnd)
    else:
        a = n // 2
        lines = synth_code(a, rnd) + synth_math(n - a, rnd)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
