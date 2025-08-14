# lldc/scripts/rd_collect_and_plot.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
import math

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent if (HERE.name == "scripts") else HERE
RESULTS = ROOT / "results"


def _try_load(p: Path) -> Any | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_points() -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {"PM": [], "VQ": [], "BASELINE": [], "AR": []}

    pm = _try_load(RESULTS / "pm_points.json")
    if isinstance(pm, list):
        for d in pm:
            d["method"] = "PM"
        out["PM"] = pm

    vq = _try_load(RESULTS / "vq_points.json")
    if isinstance(vq, list):
        for d in vq:
            d["method"] = "VQ"
        out["VQ"] = vq

    for candidate in [
        "baselines.json",
        "lossless_baselines.json",
        "compress_baselines.json",
    ]:
        j = _try_load(RESULTS / candidate)
        if isinstance(j, list):
            for d in j:
                d["method"] = d.get("method", "BASELINE")
                d["label"] = d.get("label") or d.get("name") or d["method"]
            out["BASELINE"].extend(j)

    ar = _try_load(RESULTS / "ar_points.json")
    if isinstance(ar, list):
        for d in ar:
            d["method"] = "AR"
        out["AR"] = ar

    return out


def _find_entropy_line() -> float | None:
    for fn in ["dataset_stats.json", "data_stats.json"]:
        p = RESULTS / fn
        j = _try_load(p)
        if isinstance(j, dict):
            for k in [
                "h_infty_bits_per_char",
                "H_infty_bits_per_char",
                "H_bits_per_char",
            ]:
                if k in j and isinstance(j[k], (int, float)):
                    return float(j[k])
    return None


def _extract_xy(
    points: List[Dict], metric: str
) -> Tuple[List[float], List[float], List[str]]:
    X, Y, labels = [], [], []
    for d in points:
        bpc = d.get("bpc")
        if bpc is None:
            continue
        if metric == "charF":
            dist = 1.0 - float(d.get("charF_mean", d.get("char_fidelity", 0.0)))
        elif metric == "chrf":
            dist = 1.0 - float(d.get("chrf_mean", d.get("chrf", 0.0)))
        elif metric == "bertscore_f1":
            dist = 1.0 - float(d.get("bertscore_f1_mean", d.get("bertscore_f1", 0.0)))
        elif metric == "sem_span":
            dist = 1.0 - float(
                d.get("sem_span_fid_mean", d.get("semantic_span_fid", 0.0))
            )
        else:
            dist = 1.0 - float(d.get("charF_mean", d.get("char_fidelity", 0.0)))
        X.append(float(bpc))
        Y.append(float(dist))
        lbl = d.get("label") or str(d.get("codebook_size", "")) or ""
        labels.append(lbl)
    return X, Y, labels


def plot_rd(metric: str = "charF", out_png: Path | None = None) -> Path:
    pts = _load_points()
    ent = _find_entropy_line()
    fig, ax = plt.subplots()
    for name, color, marker in [("PM", None, "o"), ("VQ", None, "s")]:
        X, Y, _ = _extract_xy(pts[name], metric)
        if X:
            ax.plot(X, Y, marker=marker, linestyle="-", label=name)

    for d in pts["BASELINE"]:
        bpc = d.get("bpc")
        if bpc is None:
            continue
        ax.scatter(
            [float(bpc)],
            [0.0],
            marker="^",
            label=d.get("label", d.get("method", "baseline")),
        )

    X, Y, _ = _extract_xy(pts["AR"], metric)
    if X:
        ax.scatter(X, Y, marker="x", label="AR")
    ax.set_xlabel("Bits per Character (BPC)")
    ax.set_ylabel(f"Distortion (1 - {metric})")
    ax.set_title("Rate–Distortion (all methods)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if ent is not None:
        ax.axvline(ent, linestyle=":", label=f"entropy ~ {ent:.3f} bpc")
    handles, labels = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        H.append(h)
        L.append(l)
    ax.legend(H, L, loc="best")
    out_png = out_png or (RESULTS / f"rd_plot_{metric}.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metric",
        default="charF",
        choices=["charF", "chrf", "bertscore_f1", "sem_span"],
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    p = plot_rd(metric=args.metric, out_png=out)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
