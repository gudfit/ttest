from __future__ import annotations
from pathlib import Path
import json
import argparse
import matplotlib.pyplot as plt


def _avg_bpc(p):
    if not isinstance(p, list) or not p:
        return None
    return sum(d.get("bpc", 0.0) for d in p) / len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/subsets")
    args = ap.parse_args()
    root = Path(args.root)

    sizes, pm_bpc, vq_bpc = [], [], []
    for d in sorted(
        [p for p in root.iterdir() if p.is_dir()], key=lambda x: int(x.name)
    ):
        pm = (
            json.loads((d / "pm_points.json").read_text())
            if (d / "pm_points.json").exists()
            else []
        )
        vq = (
            json.loads((d / "vq_points.json").read_text())
            if (d / "vq_points.json").exists()
            else []
        )
        apm = _avg_bpc(pm)
        avq = _avg_bpc(vq)
        if apm is None or avq is None:
            continue
        sizes.append(int(d.name))
        pm_bpc.append(apm)
        vq_bpc.append(avq)

    if not sizes:
        print("No subset results found.")
        return

    plt.figure()
    plt.plot(sizes, pm_bpc, marker="o", label="PM (avg BPC)")
    plt.plot(sizes, vq_bpc, marker="s", label="VQ (avg BPC)")
    plt.xlabel("Dataset size (docs)")
    plt.ylabel("Average BPC")
    plt.title("Crossover vs dataset size")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out = Path("results/crossover_vs_size.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
