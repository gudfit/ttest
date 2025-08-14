# lldc/scripts/evaluate_all.py
from __future__ import annotations
import json, re, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, DefaultDict
from collections import defaultdict

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.utils import wandb_log
from lldc.metrics.fidelity import character_level_fidelity
from lldc.eval.perplexity import ar_bpc as compute_ar_bpc
from lldc.baselines.kenlm_subsample import kenlm_ngram_bpc

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
    chrf: float | None
    berts_f1: float | None
    sem_span_fid: float | None
    cpu_decode_ms: float | None
    n_docs: int


def _safe_char_fidelity(orig: str, recon: str) -> float:
    return character_level_fidelity(orig, recon) / 100.0


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
            try:
                mask_rate = float(m.group(1))
            except Exception:
                mask_rate = None
        k = _K_PAT.search(sub.name)
        if k:
            try:
                codebook_K = int(k.group(1))
            except Exception:
                codebook_K = None

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
        chrf: List[float] = []
        berts: List[float] = []
        sems: List[float] = []
        decms: List[float] = []
        n_docs = 0
        saw_bit_fields = False

        with recons.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    j = json.loads(ln)
                except Exception:
                    continue
                n_docs += 1
                pb = int(j.get("position_bits", 0))
                tb = int(j.get("token_bits", 0))
                if pb or tb:
                    saw_bit_fields = True
                total_bits += pb + tb
                orig = j.get("original", "") or ""
                recon = j.get("reconstruction", "") or ""
                total_chars += max(1, int(j.get("orig_chars", len(orig))))
                if "char_fidelity" in j:
                    fids.append(float(j["char_fidelity"]) / 100.0)
                else:
                    fids.append(_safe_char_fidelity(orig, recon))
                if "chrf" in j:
                    chrf.append(float(j["chrf"]) / 100.0)
                if "bertscore_f1" in j:
                    berts.append(float(j["bertscore_f1"]) / 100.0)
                if "semantic_span_fid" in j:
                    sems.append(float(j["semantic_span_fid"]) / 100.0)
                if "cpu_decode_ms" in j:
                    decms.append(float(j["cpu_decode_ms"]))

        if total_chars == 0 or not saw_bit_fields:
            continue
        bpc = total_bits / float(total_chars)
        fid = float(sum(fids) / max(1, len(fids))) if fids else None
        c = float(sum(chrf) / max(1, len(chrf))) if chrf else None
        b = float(sum(berts) / max(1, len(berts))) if berts else None
        s = float(sum(sems) / max(1, len(sems))) if sems else None
        dms = float(sum(decms) / max(1, len(decms))) if decms else None

        points.append(
            RDPoint(
                method=method,
                model=model,
                mask_rate=mask_rate,
                codebook_K=codebook_K,
                bpc=bpc,
                fidelity=fid,
                chrf=c,
                berts_f1=b,
                sem_span_fid=s,
                cpu_decode_ms=dms,
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
    (out_dir / "pm_points.json").write_text(
        json.dumps([p for p in rd_points if p.get("method") == "PM"], indent=2)
    )
    (out_dir / "vq_points.json").write_text(
        json.dumps([p for p in rd_points if p.get("method") == "VQ"], indent=2)
    )


def _aggregate_mean_std(points: List[RDPoint]) -> Dict[str, Dict[str, float]]:
    """
    Group by "key": PM → (strategy, mask_rate) if available, else method only.
               VQ → (codebook_K) if available, else method only.
    Compute mean±std for bpc and fidelity-like metrics.
    """
    buckets: DefaultDict[str, List[RDPoint]] = defaultdict(list)
    for p in points:
        if p.method == "PM" and p.mask_rate is not None:
            key = f"PM@mask{p.mask_rate:.2f}"
        elif p.method == "VQ" and p.codebook_K is not None:
            key = f"VQ@K{p.codebook_K}"
        else:
            key = p.method
        buckets[key].append(p)

    def _safe_stats(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return (float("nan"), float("nan"))
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
        return (m, math.sqrt(v))

    summary: Dict[str, Dict[str, float]] = {}
    for key, arr in buckets.items():
        bpcs = [p.bpc for p in arr]
        fids = [p.fidelity for p in arr if p.fidelity is not None]
        chrf = [p.chrf for p in arr if p.chrf is not None]
        bsf = [p.berts_f1 for p in arr if p.berts_f1 is not None]
        sem = [p.sem_span_fid for p in arr if p.sem_span_fid is not None]
        dec = [p.cpu_decode_ms for p in arr if p.cpu_decode_ms is not None]
        bpc_m, bpc_s = _safe_stats(bpcs)
        fid_m, fid_s = _safe_stats(fids) if fids else (float("nan"), float("nan"))
        ch_m, ch_s = _safe_stats(chrf) if chrf else (float("nan"), float("nan"))
        bs_m, bs_s = _safe_stats(bsf) if bsf else (float("nan"), float("nan"))
        se_m, se_s = _safe_stats(sem) if sem else (float("nan"), float("nan"))
        dc_m, dc_s = _safe_stats(dec) if dec else (float("nan"), float("nan"))
        summary[key] = {
            "bpc_mean": bpc_m,
            "bpc_std": bpc_s,
            "charF_mean": fid_m,
            "charF_std": fid_s,
            "chrf_mean": ch_m,
            "chrf_std": ch_s,
            "bertscore_f1_mean": bs_m,
            "bertscore_f1_std": bs_s,
            "sem_span_fid_mean": se_m,
            "sem_span_fid_std": se_s,
            "cpu_decode_ms_mean": dc_m,
            "cpu_decode_ms_std": dc_s,
            "num_runs": len(arr),
        }
    return summary


def _compute_base_chars_from_payloads(payload_root: Path) -> int:
    total_chars = 0
    for p in payload_root.glob("*/recons.jsonl"):
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    j = json.loads(ln)
                except Exception:
                    continue
                total_chars += int(j.get("orig_chars", len(j.get("original", ""))))
    return total_chars


def _amortised_bpc(
    bpc_runtime: float,
    static_bits: int,
    base_chars: int,
    copies: int = 1,
    scale: int = 1,
) -> float:
    denom = max(1, base_chars * copies * scale)
    return static_bits / denom + bpc_runtime


def _export_amortised_curves(
    out_dir: Path, methods: Dict[str, float], static_bits: int, base_chars: int
) -> None:
    import numpy as np

    Ns = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    scales = [1, 2, 4, 8, 16, 32]
    curves = []
    for name, bpc in methods.items():
        for n in Ns:
            for s in scales:
                curves.append(
                    {
                        "method": name,
                        "N_copies": n,
                        "scale": s,
                        "amortised_bpc": _amortised_bpc(
                            bpc, static_bits, base_chars, n, s
                        ),
                    }
                )
    (out_dir / "amortised_bpc_curves.json").write_text(json.dumps(curves, indent=2))
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        for name, bpc in methods.items():
            y = [_amortised_bpc(bpc, static_bits, base_chars, n, 1) for n in Ns]
            plt.plot(Ns, y, label=name, marker="o")
        plt.xscale("log")
        plt.xlabel("N_copies (log scale)")
        plt.ylabel("Amortised BPC (scale=1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "amortised_bpc.png", dpi=160)
        plt.close()
    except Exception:
        pass


def main():
    import hydra

    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        paths = Paths().ensure()

        rd_points_objects = _scan_payload_dir(paths.payloads)
        rd_points = []
        for p in rd_points_objects:
            rd_points.append(
                {
                    "method": p.method,
                    "model": p.model,
                    "mask_rate": p.mask_rate,
                    "codebook_K": p.codebook_K,
                    "bpc": p.bpc,
                    "charF_mean": p.fidelity,
                    "chrf_mean": p.chrf,
                    "bertscore_f1_mean": p.berts_f1,
                    "sem_span_fid_mean": p.sem_span_fid,
                    "cpu_decode_ms_mean": p.cpu_decode_ms,
                    "n_docs": p.n_docs,
                }
            )

        static = _compute_static_bits(paths, cfg)
        _write_static_and_rd_exports(paths, static, rd_points)

        agg = _aggregate_mean_std(rd_points_objects)
        (paths.results / "rd_aggregates.json").write_text(json.dumps(agg, indent=2))

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            ds = __import__("datasets").load_dataset(
                cfg.data.source.hf_dataset, cfg.data.source.hf_config
            )
            texts = [
                ex[cfg.data.processing.text_field]
                for ex in ds[cfg.data.source.split_map.test]
            ]
            tok = AutoTokenizer.from_pretrained("gpt2-large")
            if tok.pad_token is None and tok.eos_token:
                tok.pad_token = tok.eos_token
            m = AutoModelForCausalLM.from_pretrained("gpt2-large")
            ar_bpc_val = compute_ar_bpc(
                m,
                tok,
                texts,
                device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            )
            (paths.results / "ar_baseline.json").write_text(
                json.dumps({"model": "gpt2-large", "bpc": ar_bpc_val}, indent=2)
            )
        except Exception as e:
            (paths.results / "ar_baseline.json").write_text(
                json.dumps({"error": str(e)}, indent=2)
            )

        try:
            ds = __import__("datasets").load_dataset(
                cfg.data.source.hf_dataset, cfg.data.source.hf_config
            )
            text_field = cfg.data.processing.text_field
            train_texts = [ex[text_field] for ex in ds[cfg.data.source.split_map.train]]
            test_texts = [ex[text_field] for ex in ds[cfg.data.source.split_map.test]]
            bpc8 = kenlm_ngram_bpc(
                train_texts,
                test_texts,
                workdir="artifacts/runs/kenlm_8gram_eval",
                order=8,
            )
            (paths.results / "ngram8_baseline.json").write_text(
                json.dumps({"order": 8, "bpc": bpc8}, indent=2)
            )
        except Exception as e:
            (paths.results / "ngram8_baseline.json").write_text(
                json.dumps({"error": str(e)}, indent=2)
            )

        baselines_file = paths.results / "baselines.json"
        if baselines_file.exists():
            log.info(f"[evaluate_all] Found external baselines -> {baselines_file}")

        methods_bpc_runtime: Dict[str, float] = {}
        for p in rd_points_objects:
            key = f"{p.method}"
            methods_bpc_runtime[key] = min(
                methods_bpc_runtime.get(key, float("inf")), p.bpc
            )
        base_chars = _compute_base_chars_from_payloads(paths.payloads)
        _export_amortised_curves(
            paths.results, methods_bpc_runtime, static["static_bits_total"], base_chars
        )

        log.info(
            f"[evaluate_all] Wrote RD points, static size, aggregates and amortised curves to {paths.results}"
        )

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
