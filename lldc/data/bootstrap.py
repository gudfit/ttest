from __future__ import annotations
from pathlib import Path
from .probes import generate_factual_probes, ProbeSpec
from .ood import ensure_ood


def ensure_data(root: Path) -> dict:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    probes = data_dir / "factual_probes.jsonl"
    ood = data_dir / "ood_texts.txt"
    if not probes.exists():
        generate_factual_probes(ProbeSpec(out_path=probes, n=800))
    if not ood.exists():
        ensure_ood(ood, n=500, mode="synthetic")
    return {"probes": probes, "ood": ood}
