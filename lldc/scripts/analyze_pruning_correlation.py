# lldc/scripts/analyze_pruning_correlation.py

from __future__ import annotations
import json, math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import re
import hydra
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.power import TTestIndPower

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths


def _float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _scan_results(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def _prune_level_from_name(name: str) -> Optional[float]:
        for pat in [
            r"pruned[_-](0\.\d+|1(?:\.0+)?)",
            r"prune[_-](0\.\d+|1(?:\.0+)?)",
            r"level(0\.\d+|1(?:\.0+)?)",
        ]:
            m = re.search(pat, name)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
        return None

    for sub in (root / "artifacts").glob("*"):
        if not sub.is_dir():
            continue
        level = _prune_level_from_name(sub.name)
        if level is None:
            for deeper in sub.glob("*"):
                if not deeper.is_dir():
                    continue
                level = _prune_level_from_name(deeper.name)
                run_root = deeper
                if level is None:
                    continue
                break
            else:
                continue
        else:
            run_root = sub

        res_dir = run_root / "results"
        rd_dir = run_root / "rd_curves"
        ar = res_dir / "abstract_reasoning.json"
        fr = res_dir / "factual_recall.json"
        pm = rd_dir / "pm_points.json"

        glue = None
        fac = None
        tcm = None
        pcm = None

        try:
            if ar.exists():
                j = json.loads(ar.read_text())
                glue = _float(j.get("macro_f1"))
        except Exception:
            pass

        try:
            if fr.exists():
                j = json.loads(fr.read_text())
                fac = _float(j.get("accuracy_bleurt"))
        except Exception:
            pass

        try:
            if pm.exists():
                arr = json.loads(pm.read_text())
                if isinstance(arr, list) and arr:
                    tcm = _float(arr[0].get("tcm_mean"))
                    pcm = _float(arr[0].get("pcm_mean"))
        except Exception:
            pass

        rows.append(
            {
                "run_dir": str(run_root),
                "pruning_level": level,
                "glue_macro_f1": glue,
                "factual_recall_bleurt": fac,
                "tcm_bits": tcm,
                "pcm_bits": pcm,
            }
        )
    return rows


def _pearson(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    x2 = x.dropna()
    y2 = y.dropna()
    df = pd.concat([x2, y2], axis=1).dropna()
    if df.shape[0] < 3:
        return float("nan"), float("nan")
    r, p = pearsonr(df.iloc[:, 0], df.iloc[:, 1])
    return float(r), float(p)


def _perform_power_analysis(effect_size=0.5, alpha=0.05, power=0.8) -> int:
    analysis = TTestIndPower()
    required_n = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    return int(math.ceil(required_n))


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: Any) -> None:
    log = setup_logging()
    paths = Paths().ensure()

    rows = _scan_results(paths.root)
    if not rows:
        log.warning("[correlation] No pruning runs found under artifacts/.")
        return

    df = pd.DataFrame(rows)
    out_dir = paths.results / "pruning_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "raw_table.tsv", sep="\t", index=False)
    required_sample_size = _perform_power_analysis(
        effect_size=0.5, alpha=0.05, power=0.8
    )
    log.info(
        f"[correlation] Required sample size for desired power (d=0.5, α=0.05, 1-β=0.8): n = {required_sample_size}"
    )

    metrics = []
    for struct_col in ["tcm_bits", "pcm_bits"]:
        for func_col in ["glue_macro_f1", "factual_recall_bleurt"]:
            r, p = _pearson(df[struct_col], df[func_col])
            actual_n = int(df[[struct_col, func_col]].dropna().shape[0])
            metrics.append(
                {
                    "x_struct": struct_col,
                    "y_func": func_col,
                    "pearson_r": r,
                    "p_value": p,
                    "n_actual": actual_n,
                    "n_required_for_power": required_sample_size,
                    "is_powered": bool(actual_n >= required_sample_size),
                }
            )

    (out_dir / "summary.json").write_text(json.dumps(metrics, indent=2))

    try:
        import matplotlib.pyplot as plt

        for m in metrics:
            x = m["x_struct"]
            y = m["y_func"]
            sub = df[[x, y]].dropna()
            if sub.shape[0] < 3:
                continue
            plt.figure()
            plt.scatter(sub[x], sub[y], marker="o")
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(
                f"{x} vs {y} (r={m['pearson_r']:.3f}, p={m['p_value']:.3g}, n={m['n_actual']}, "
                f"powered={m['is_powered']})"
            )
            plt.tight_layout()
            plt.savefig(out_dir / f"scatter_{x}_vs_{y}.png", dpi=170)
            plt.close()
    except Exception:
        pass

    log.info(f"[correlation] Wrote table + summary to {out_dir}")


if __name__ == "__main__":
    main()
