# lldc/scripts/evaluate_reconstructions.py

from __future__ import annotations
from typing import Any, List, Dict
import json
from pathlib import Path

import hydra

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.eval.hallucination import evaluate_nli_hallucination


def _load_recons(path: Path) -> tuple[List[str], List[str]]:
    originals, recons = [], []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                j = json.loads(ln)
            except Exception:
                continue
            originals.append(j.get("original", "") or "")
            recons.append(j.get("reconstruction", "") or "")
    return originals, recons


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: Any) -> None:
    log = setup_logging()
    paths = Paths().ensure()
    recon_path = Path(str(getattr(cfg, "recon_path", "")))
    if not recon_path.exists():
        cands = list((paths.payloads).glob("*/recons.jsonl"))
        if not cands:
            raise FileNotFoundError(
                "No reconstructions found. Provide cfg.recon_path=path/to/recons.jsonl."
            )
        recon_path = cands[0]
        log.info(f"[eval_recons] Using found recon file: {recon_path}")
    orig, rec = _load_recons(recon_path)
    if not orig or not rec:
        raise RuntimeError("Empty reconstructions set – nothing to evaluate.")
    res = evaluate_nli_hallucination(orig, rec)
    out = paths.results / "reconstruction_functional.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log.info(f"[eval_recons] Wrote {out}")


if __name__ == "__main__":
    main()
