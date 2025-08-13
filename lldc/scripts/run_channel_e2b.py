from __future__ import annotations
from typing import Any, List, Dict
import json
from pathlib import Path
from datasets import load_dataset
from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.scripts.channel_analysis import run_channel_analysis, KN5


def _load_recons(paths: List[Path]) -> List[Dict]:
    docs = []
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    docs.append(json.loads(line))
                except Exception:
                    pass
    return docs


def main():
    import hydra

    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        paths = Paths().ensure()
        payload_root = paths.payloads

        pm_paths = list(payload_root.glob("pm_*/recons.jsonl"))
        vq_paths = list(payload_root.glob("vq_*/recons.jsonl"))

        pm_docs = _load_recons(pm_paths)
        vq_docs = _load_recons(vq_paths)
        all_docs = pm_docs + vq_docs

        orig = [d.get("original", "") for d in all_docs]
        recon = [d.get("reconstruction", "") for d in all_docs]

        if not orig or not recon:
            raise RuntimeError(
                "[e2b] No reconstructions found in artifacts/payloads/. "
                "Run e1a or stage2_* to generate recon dumps before channel analysis."
            )

        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        train_texts = [
            ex[cfg.data.processing.text_field]
            for ex in ds[cfg.data.source.split_map.train]
        ]
        kn5 = KN5(n=5, discount=0.75, vocab_limit=50000)
        kn5.fit(train_texts)

        stats = run_channel_analysis(cfg, recon_texts=recon, orig_texts=orig, kn5=kn5)
        out = paths.rd_curves / "channel_stats.json"
        out.write_text(json.dumps(stats, indent=2))
        log.info(f"[e2b] Wrote channel stats → {out}")

    _run()


if __name__ == "__main__":
    main()
