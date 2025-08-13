# lldc/scripts/evaluate_all.py
from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.utils import wandb_log

_MASK_PAT = re.compile(r"mask(?:_|-)?(0\.\d+|1(?:\.0+)?)", re.I)
_K_PAT = re.compile(r"(?:k|codebook|K)(?:_|-)?(\d{2,6})", re.I)


@dataclass
class RDPoint:
    method: str
    model: str | None
    mask_rate: float | None
    codebook_K: int | None
    bpc: float
    fidelity: float | None
    n_docs: int


def _safe_char_fidelity(orig: str, recon: str) -> float:
    if not orig and not recon:
        return 1.0
    denom = max(len(orig), len(recon), 1)
    match = sum(1 for a, b in zip(orig, recon) if a == b)
    return match / denom


def _scan_payload_dir(payload_root: Path) -> List[RDPoint]:
    points: List[RDPoint] = []
    for sub in sorted(payload_root.glob("*")):
        if not sub.is_dir():
            continue
        method = (
            "PM"
            if sub.name.lower().startswith("pm_")
            else "VQ" if sub.name.lower().startswith("vq_") else None
        )
        if method is None:
            continue

        recons = sub / "recons.jsonl"
        if not recons.exists():
            continue

        mask_rate = None
        codebook_K = None
        m = _MASK_PAT.search(sub.name)
        if m:
            mask_rate = float(m.group(1))
        k = _K_PAT.search(sub.name)
        if k:
            codebook_K = int(k.group(1))

        model = None
        parts = sub.name.split("_")
        if len(parts) >= 2:
            cand = parts[1]
            cand = (
                cand.replace("mask", "").replace("k", "").replace("-", "-").strip("-_")
            )
            model = cand if cand else None

        total_bits = 0
        total_chars = 0
        fids: List[float] = []
        n_docs = 0

        with recons.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    j = json.loads(ln)
                except Exception:
                    continue
                n_docs += 1
                total_bits += int(j.get("payload_bits", 0))
                orig = j.get("original", "") or ""
                recon = j.get("reconstruction", "") or ""
                total_chars += max(1, int(j.get("orig_chars", len(orig))))
                if "char_fidelity" in j:
                    fids.append(float(j["char_fidelity"]))
                else:
                    fids.append(_safe_char_fidelity(orig, recon))

        if total_chars == 0:
            continue
        bpc = total_bits / float(total_chars)
        fid = float(sum(fids) / max(1, len(fids))) if fids else None

        points.append(
            RDPoint(
                method=method,
                model=model,
                mask_rate=mask_rate,
                codebook_K=codebook_K,
                bpc=bpc,
                fidelity=fid,
                n_docs=n_docs,
            )
        )
    return points


_MODEL_PATTERNS = ("*.bin", "*.pt", "*.safetensors")
_CODEBOOK_PATTERNS = (
    "*codebook*.pt",
    "*codebook*.bin",
    "*codebook*.safetensors",
    "*vq*codebook*.*",
    "*_codebook*.*",
)
_GRU_PATTERNS = (
    "*gru*.pt",
    "*gru*.bin",
    "*gru*.safetensors",
    "*index_lm*.pt",
    "*index_lm*.bin",
    "*index_lm*.safetensors",
)


def _files_matching(root: Path, patterns: Tuple[str, ...]) -> Set[Path]:
    out: Set[Path] = set()
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file():
                out.add(p.resolve())
    return out


def _sum_bytes(files: Set[Path]) -> int:
    total = 0
    for p in files:
        try:
            total += p.stat().st_size
        except Exception:
            pass
    return total


def _compute_static_bits(paths: Paths, cfg: Any) -> Dict[str, int]:
    ckpt_root = paths.checkpoints

    codebook_files = _files_matching(ckpt_root, _CODEBOOK_PATTERNS)
    gru_files = _files_matching(ckpt_root, _GRU_PATTERNS)
    model_files = _files_matching(ckpt_root, _MODEL_PATTERNS)

    model_only = model_files.difference(codebook_files).difference(gru_files)

    model_bits = _sum_bytes(model_only) * 8
    codebook_bits = _sum_bytes(codebook_files) * 8
    gru_bits = _sum_bytes(gru_files) * 8
    total_bits = model_bits + codebook_bits + gru_bits

    override = int(getattr(getattr(cfg, "experiment", {}), "static_bits_override", 0))
    if total_bits == 0 and override > 0:
        total_bits = override

    return {
        "static_bits_total": int(total_bits),
        "model_bits": int(model_bits),
        "codebook_bits": int(codebook_bits),
        "gru_bits": int(gru_bits),
    }


def _write_static_and_rd_exports(
    paths: Paths, static_bits: Dict[str, int], rd_points: list[dict]
) -> None:
    out_dir = paths.results
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "static_size.json").write_text(
        json.dumps(
            {
                "static_bits_total": static_bits["static_bits_total"],
                "breakdown": {
                    "model_bits": static_bits["model_bits"],
                    "codebook_bits": static_bits["codebook_bits"],
                    "gru_bits": static_bits["gru_bits"],
                },
            },
            indent=2,
        )
    )

    (out_dir / "pm_points.json").write_text(json.dumps(rd_points, indent=2))


def main():
    import hydra

    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        paths = Paths().ensure()

        rd_points_objects = _scan_payload_dir(paths.payloads)
        rd_points = [p.__dict__ for p in rd_points_objects]

        static = _compute_static_bits(paths, cfg)
        _write_static_and_rd_exports(paths, static, rd_points)

        log.info(f"[evaluate_all] Wrote RD points and static size to {paths.results}")

        try:
            run = wandb_log.start(
                cfg, run_name="evaluate-all", tags=["evaluate", "export"]
            )
            wandb_log.log(
                {
                    "static/static_bits_total": static["static_bits_total"],
                    "static/model_bits": static["model_bits"],
                    "static/codebook_bits": static["codebook_bits"],
                    "static/gru_bits": static["gru_bits"],
                    "rd/num_points": len(rd_points),
                }
            )
            wandb_log.finish()
        except Exception:
            pass

    _run()


if __name__ == "__main__":
    main()
