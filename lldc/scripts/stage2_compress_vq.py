# lldc/scripts/stage2_compress_vq.py

from __future__ import annotations
from typing import Any, List, Dict
import json
import math  # <-- needed for math.inf
from pathlib import Path
import time
import hydra
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from lldc.utils.paths import Paths
from lldc.utils.logging import setup_logging
from lldc.models.vq.vq_trainer import (
    train_vq_joint,
    encode_indices,
    train_index_lm,
    cross_entropy_bits_index_stream,
)


def _cfg_get(cfg: Any, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _limit(n: int | None, default: int) -> int:
    try:
        if n is None:
            return default
        return int(n)
    except Exception:
        return default


def _resolve_base_model_name(cfg: Any, paths: Paths) -> str:
    ckpt = _cfg_get(cfg, "model_ckpt", None)
    if ckpt and Path(str(ckpt)).exists():
        return str(ckpt)
    return cfg.model.pretrained_name


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: Any) -> None:
    log = setup_logging()
    paths = Paths().ensure()
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    split_map = cfg.data.source.split_map
    text_field = cfg.data.processing.text_field
    train_split = ds[split_map.train].select(range(10000))
    test_split = ds[split_map.test].select(range(1000))

    layer_after = int(_cfg_get(cfg, "stage2.vq.bottleneck.layer_after", 6))
    beta = float(_cfg_get(cfg, "stage2.vq.bottleneck.commitment_beta", 0.25))
    K = int(_cfg_get(cfg, "stage2.vq.codebook_sizes.0", 256))
    idx_hidden = int(_cfg_get(cfg, "stage2.vq.index_lm.hidden_size", 512))
    idx_layers = int(_cfg_get(cfg, "stage2.vq.index_lm.layers", 1))
    idx_epochs = int(_cfg_get(cfg, "stage2.vq.index_lm.epochs", 2))

    base_model_name = _resolve_base_model_name(cfg, paths)
    if Path(base_model_name).exists():
        log.info(f"[stage2_vq] Using base model from checkpoint: {base_model_name}")
    else:
        log.info(f"[stage2_vq] Using base model from hub: {base_model_name}")

    model_vq, tok = train_vq_joint(
        base_model_name=base_model_name,
        dataset_name=cfg.data.source.hf_dataset,
        dataset_config=cfg.data.source.hf_config,
        text_field=text_field,
        max_length=cfg.data.processing.max_length,
        layer_after=layer_after,
        codebook_size=K,
        lr=5e-5,
        epochs=idx_epochs,
        beta=beta,
    )
    model_vq.eval().to(device)

    max_train = _limit(
        getattr(getattr(cfg, "data", {}), "limits", {}).get("max_train_samples"), 10000
    )
    idx_train: List[List[int]] = []
    for ex in train_split.select(range(min(max_train, len(train_split)))):
        txt = ex.get(text_field) or ""
        if not txt:
            continue
        ids = tok(
            txt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.data.processing.max_length,
            add_special_tokens=False,
        )["input_ids"].to(device)
        seq_idx = encode_indices(model_vq, ids)[0].tolist()
        if seq_idx:
            idx_train.append(seq_idx)

    lm = (
        train_index_lm(
            idx_train,
            K=K,
            hidden=idx_hidden,
            layers=idx_layers,
            epochs=idx_epochs,
        )
        .to(device)
        .eval()
    )

    max_eval = _limit(
        getattr(getattr(cfg, "data", {}), "limits", {}).get("max_eval_samples"), 2000
    )

    payload_dir = (
        paths.payloads / f"vq_{cfg.model.pretrained_name.replace('/', '-')}_K{K}"
    )
    payload_dir.mkdir(parents=True, exist_ok=True)
    recons_path = payload_dir / "recons.jsonl"

    total_bits = 0.0
    total_tokens = 0
    total_chars = 0
    n_docs = 0

    with recons_path.open("w", encoding="utf-8") as fout:
        for ex in test_split.select(range(min(max_eval, len(test_split)))):
            txt = ex.get(text_field) or ""
            if not txt:
                continue
            ids = tok(
                txt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.data.processing.max_length,
                add_special_tokens=False,
            )["input_ids"].to(device)
            seq_idx = encode_indices(model_vq, ids)[0].tolist()
            bits = cross_entropy_bits_index_stream(lm, seq_idx)
            total_bits += bits
            total_tokens += max(0, len(seq_idx) - 1)
            total_chars += len(txt)
            n_docs += 1
            with torch.no_grad():
                idx_t = torch.tensor(
                    seq_idx, dtype=torch.long, device=device
                ).unsqueeze(0)
                toks_pred = model_vq.decode_from_indices(idx_t)[0].tolist()
            recon = tok.decode(toks_pred, skip_special_tokens=True)
            doc_rec = {
                "original": txt,
                "reconstruction": recon,
                "orig_chars": len(txt),
                "token_bits": int(round(bits)),
                "position_bits": 0,
                "index_len": int(len(seq_idx)),
                "kept_tokens": 0,
            }
            fout.write(json.dumps(doc_rec) + "\n")

    bpt = (total_bits / max(1, total_tokens)) if total_tokens > 0 else None
    bpc = total_bits / max(1, total_chars) if total_chars > 0 else math.inf

    summary = {
        "method": "VQ",
        "model": cfg.model.pretrained_name,
        "codebook_K": K,
        "index_bits": float(total_bits),
        "tokens": int(total_tokens),
        "chars": int(total_chars),
        "bpt": (float(bpt) if bpt is not None else None),
        "bpc": float(bpc),
        "n_docs": int(n_docs),
        "notes": "Index LM trained on TRAIN split indices only; evaluated on TEST indices.",
    }
    (payload_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"[stage2_vq] Wrote {recons_path} and summary: bpc={bpc:.6f}, bpt={bpt}")


if __name__ == "__main__":
    main()

