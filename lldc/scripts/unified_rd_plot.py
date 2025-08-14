# lldc/scripts/unified_rd_plot.py
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

from lldc.metrics.fidelity import character_level_fidelity
from lldc.metrics.pm_bpt import pm_bpt_bpc_from_fraction, pm_bpt_from_fraction

try:
    import yaml
except Exception:
    yaml = None


def _read_lines(p: Path) -> List[str]:
    with p.open("r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _pairwise_charF(orig: List[str], recon: List[str]) -> float:
    return _mean([character_level_fidelity(o, r) for o, r in zip(orig, recon)])


def _load_points_from_config(
    cfg: Dict, defaults: Dict
) -> Tuple[List[Dict], Dict[str, str]]:
    series_colors: Dict[str, str] = {}
    for s in cfg.get("series", []):
        nm = s["name"]
        if "color" in s:
            series_colors[nm] = s["color"]

    pts = []
    for p in cfg.get("points", []):
        point = {**defaults, **p}
        pts.append(point)
    return pts, series_colors


def build_rd_points(
    points_cfg: List[Dict],
) -> Tuple[Dict[str, List[Tuple[float, float, str, Optional[float]]]], List[str]]:
    series_points: Dict[str, List[Tuple[float, float, str, Optional[float]]]] = {}
    series_order: List[str] = []

    for p in points_cfg:
        series = p["series"]
        label = p.get("label", "")
        orig = _read_lines(Path(p["orig"]))
        recon = _read_lines(Path(p["recon"]))
        if len(orig) != len(recon):
            raise ValueError(
                f"Length mismatch for {label}: {len(orig)} vs {len(recon)}"
            )

        charF = _pairwise_charF(orig, recon)
        distortion = 1.0 - charF

        rate_bpc: Optional[float] = p.get("rate_bpc", None)
        bpt_to_annotate: Optional[float] = None

        if rate_bpc is None:
            keep_fraction = p.get("keep_fraction", None)
            vocab_size = p.get("vocab_size", None)
            if keep_fraction is None or vocab_size is None:
                raise ValueError(
                    f"Point '{label}' needs either `rate_bpc` or both `keep_fraction` and `vocab_size`."
                )
            entropy_coded = bool(p.get("entropy_coded", False))
            tokenizer_name = p.get("tokenizer", None)
            bpt, bpc, tpc = pm_bpt_bpc_from_fraction(
                orig,
                float(keep_fraction),
                int(vocab_size),
                tokenizer_name,
                entropy_coded=entropy_coded,
            )
            rate_bpc = bpc
            if p.get("annotate_bpt", True):
                bpt_to_annotate = bpt

        tup = (float(rate_bpc), float(distortion), label, bpt_to_annotate)
        series_points.setdefault(series, []).append(tup)
        if series not in series_order:
            series_order.append(series)

    for s in series_points:
        series_points[s].sort(key=lambda x: x[0])

    return series_points, series_order


def plot_unified_rd(
    series_points: Dict[str, List[Tuple[float, float, str, Optional[float]]]],
    series_order: List[str],
    series_colors: Dict[str, str],
    out_path: Path,
    title: str = "Unified Rate–Distortion",
):
    plt.figure(figsize=(8, 5.5))
    for s in series_order:
        pts = series_points[s]
        xs = [x for (x, _, _, _) in pts]
        ys = [y for (_, y, _, _) in pts]
        lbls = [lbl for (_, _, lbl, _) in pts]
        bpts = [bp for (_, _, _, bp) in pts]
        color = series_colors.get(s, None)

        if color is None:
            lh = plt.plot(xs, ys, marker="o", label=s)
        else:
            lh = plt.plot(xs, ys, marker="o", label=s, color=color)

        for x, y, lbl, bpt in pts:
            suffix = f" [BPT={bpt:.3f}]" if (bpt is not None) else ""
            plt.annotate(
                f"{lbl}{suffix}",
                (x, y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

    plt.xlabel("Rate (bits per character, BPC)")
    plt.ylabel("Distortion (1 − charF)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png = out_path.with_suffix(".png")
    pdf = out_path.with_suffix(".pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=160)
    plt.savefig(pdf)
    print(f"Saved: {png} and {pdf}")


def main():
    ap = argparse.ArgumentParser(
        description="Unified RD plotter (BPC vs 1−charF) with PM BPT annotations."
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML/JSON with `series` and `points`.",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path prefix, e.g. results/unified_rd",
    )
    ap.add_argument("--title", type=str, default="Unified Rate–Distortion")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError(
                "pyyaml not installed. `pip install pyyaml` or provide JSON config."
            )
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    defaults = cfg.get("_defaults", {})
    points_cfg, series_colors = _load_points_from_config(cfg, defaults)
    series_points, series_order = build_rd_points(points_cfg)
    plot_unified_rd(
        series_points, series_order, series_colors, Path(args.out), title=args.title
    )


if __name__ == "__main__":
    main()
