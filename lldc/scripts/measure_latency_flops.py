# lldc/scripts/measure_latency_flops.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import json
import time
import math

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from datasets import load_dataset

from lldc.utils.paths import Paths
from lldc.utils.logging import setup_logging
from lldc.metrics.latency_rate import (
    aggregate_cpu_decode_ms_from_payloads,
    estimate_transformer_flops_per_seq,
)


@dataclass
class _MeasureCfg:
    max_examples: int = 128
    batch_size: int = 8
    max_length: Optional[int] = None


def _arch_from_cfg(cfg: Any) -> str:
    arch = getattr(cfg.model, "arch", None)
    if arch:
        return str(arch)
    name = str(getattr(cfg.model, "pretrained_name", "")).lower()
    ar_markers = ("gpt", "llama", "mistral", "opt", "phi", "qwen", "glm", "mpt")
    return "ar" if any(m in name for m in ar_markers) else "mlm"


@torch.no_grad()
def _time_encode_gpu(
    model, tok, texts: List[str], batch_size: int, max_length: Optional[int]
) -> Dict[str, float]:
    device = next(model.parameters()).device
    t_enc: List[float] = []

    def batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for batch in batches(texts, batch_size):
        enc = tok(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length or getattr(tok, "model_max_length", 1024),
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t0 = time.perf_counter()
        out = model(**enc)
        _ = out.logits if hasattr(out, "logits") else out
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) * 1000.0
        t_enc.append(dt)

    if not t_enc:
        return {"encode_ms_mean": 0.0, "encode_ms_std": 0.0}

    mean = sum(t_enc) / len(t_enc)
    var = sum((x - mean) ** 2 for x in t_enc) / max(1, len(t_enc) - 1)
    return {"encode_ms_mean": float(mean), "encode_ms_std": float(math.sqrt(var))}


def main():
    import hydra

    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        paths = Paths().ensure()

        mc = _MeasureCfg(
            max_examples=int(
                getattr(getattr(cfg, "evaluation", {}), "latency_samples", 128) or 128
            ),
            batch_size=int(getattr(cfg.data.loader, "batch_size", 8) or 8),
            max_length=getattr(cfg.data.processing, "max_length", None),
        )

        arch = _arch_from_cfg(cfg)
        name = cfg.model.pretrained_name
        if arch == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(name)
        else:
            model = AutoModelForCausalLM.from_pretrained(name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None and tok.eos_token:
            tok.pad_token = tok.eos_token

        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        tf = cfg.data.processing.text_field
        test = ds[cfg.data.source.split_map.test]
        texts: List[str] = []
        for i in range(min(mc.max_examples, len(test))):
            texts.append(test[i][tf])

        enc_stats = _time_encode_gpu(model, tok, texts, mc.batch_size, mc.max_length)

        cpu_dec = aggregate_cpu_decode_ms_from_payloads(paths.payloads)

        seq_len = mc.max_length or getattr(tok, "model_max_length", 1024)
        flops_seq = estimate_transformer_flops_per_seq(model, int(seq_len))

        out = {
            "model": name,
            "arch": arch,
            "device": device,
            "encode_gpu": enc_stats,
            "decode_cpu_aggregate": cpu_dec,
            "estimate_flops_per_seq": float(flops_seq),
            "seq_len_assumed": int(seq_len),
            "notes": (
                "FLOPs includes 2·L²·d self-attention mixing term + projections/MLP; "
                "encode times from forward() batches."
            ),
        }

        out_path = paths.results / "latency_flops.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        log.info(f"[latency/flops] Wrote {out_path}")

    _run()


if __name__ == "__main__":
    main()
