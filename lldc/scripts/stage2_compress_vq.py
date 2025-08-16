# lldc/scripts/stage2_compress_vq.py

from __future__ import annotations
from typing import Any, List, Dict
import json
from pathlib import Path
import time
import math
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
    _IndexGRULM,
    _build_vq_wrapper,
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

    base_model_name = _resolve_base_model_name(cfg, paths)
    K = int(_cfg_get(cfg, "stage2.vq.codebook_sizes.0", 256))
    exp_name = str(getattr(getattr(cfg, "experiment", {}), "name", "default"))
    force_retrain = bool(_cfg_get(cfg, "stage2.vq.force_retrain", False))

    safe_model_name = Path(base_model_name).name if Path(base_model_name).exists() else base_model_name.replace("/", "-")
    cache_dir = paths.checkpoints / f"vq_cache/{exp_name}/{safe_model_name}_K{K}"
    vq_model_path = cache_dir / "vq_model.pt"
    idx_lm_path = cache_dir / "idx_lm.pt"
    tok_path = cache_dir
    
    if not force_retrain and vq_model_path.exists() and idx_lm_path.exists() and (tok_path / "tokenizer.json").exists():
        log.info(f"[stage2_vq] Loading cached VQ model and index LM from {cache_dir}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model_vq = _build_vq_wrapper(base_model, layer_after=int(_cfg_get(cfg, "stage2.vq.bottleneck.layer_after", 6)), codebook_size=K, beta=float(_cfg_get(cfg, "stage2.vq.bottleneck.commitment_beta", 0.25)))
        model_vq.load_state_dict(torch.load(vq_model_path, map_location=device))
        model_vq.to(device).eval()
        tok = AutoTokenizer.from_pretrained(tok_path)
        lm = _IndexGRULM(K=K, hidden=int(_cfg_get(cfg, "stage2.vq.index_lm.hidden_size", 512)), layers=int(_cfg_get(cfg, "stage2.vq.index_lm.layers", 1)))
        lm.load_state_dict(torch.load(idx_lm_path, map_location=device))
        lm.to(device).eval()
    else:
        log.info(f"[stage2_vq] No valid cache found or retraining forced. Training models...")
        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        split_map = cfg.data.source.split_map
        text_field = cfg.data.processing.text_field
        train_split = ds[split_map.train]
        
        layer_after = int(_cfg_get(cfg, "stage2.vq.bottleneck.layer_after", 6))
        beta = float(_cfg_get(cfg, "stage2.vq.bottleneck.commitment_beta", 0.25))
        idx_hidden = int(_cfg_get(cfg, "stage2.vq.index_lm.hidden_size", 512))
        idx_layers = int(_cfg_get(cfg, "stage2.vq.index_lm.layers", 1))
        idx_epochs = int(_cfg_get(cfg, "stage2.vq.index_lm.epochs", 2))

        joint_train_epochs = 3
        if "e2a_pruning" in exp_name:
            recovery_epochs = None
            if Path(base_model_name).exists() and Path(base_model_name).is_dir():
                hparams_path = Path(base_model_name) / "recovery_hparams.json"
                if hparams_path.exists():
                    try:
                        hparams = json.loads(hparams_path.read_text())
                        recovery_epochs = hparams.get("epochs")
                    except Exception as e:
                        log.warning(f"Could not read recovery params from {hparams_path}: {e}")
            if recovery_epochs is not None:
                joint_train_epochs = int(recovery_epochs)
                log.info(f"[stage2_vq] Matched epochs to recovery checkpoint: {joint_train_epochs} epochs.")
            else:
                joint_train_epochs = 1
                log.warning(f"[stage2_vq] Could not find recovery params for {base_model_name}. Falling back to {joint_train_epochs} epoch for e2a experiment fairness.")
        else:
            log.info(f"[stage2_vq] Using default {joint_train_epochs} epochs for joint training (fair for e1a).")

        max_train_for_vq = _limit(_cfg_get(cfg, "data.limits.max_train_samples"), 10000)
        model_vq, tok = train_vq_joint(
            base_model_name=base_model_name,
            dataset_name=cfg.data.source.hf_dataset,
            dataset_config=cfg.data.source.hf_config,
            text_field=text_field,
            max_length=cfg.data.processing.max_length,
            layer_after=layer_after,
            codebook_size=K,
            lr=5e-5,
            epochs=joint_train_epochs,
            beta=beta,
            max_train_samples=max_train_for_vq,
        )
        model_vq.eval().to(device)

        max_train_for_idx = _limit(getattr(getattr(cfg, "data", {}), "limits", {}).get("max_train_samples"), 10000)
        if len(train_split) > max_train_for_idx:
            train_split = train_split.select(range(max_train_for_idx))
        idx_train: List[List[int]] = []
        for ex in train_split:
            txt = ex.get(text_field) or ""
            if not txt.strip(): continue
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=cfg.data.processing.max_length, add_special_tokens=False)["input_ids"].to(device)
            if ids.numel() == 0 or ids.shape[-1] == 0: continue
            seq_idx = encode_indices(model_vq, ids)[0].tolist()
            if seq_idx: idx_train.append(seq_idx)
        
        lm = train_index_lm(idx_train, K=K, hidden=idx_hidden, layers=idx_layers, epochs=idx_epochs).to(device).eval()
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_vq.state_dict(), vq_model_path)
        torch.save(lm.state_dict(), idx_lm_path)
        tok.save_pretrained(tok_path)
        log.info(f"[stage2_vq] Saved VQ model and index LM to cache: {cache_dir}")

    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    test_split = ds[cfg.data.source.split_map.test]
    text_field = cfg.data.processing.text_field
    max_eval = _limit(getattr(getattr(cfg, "data", {}), "limits", {}).get("max_eval_samples"), 2000)
    if len(test_split) > max_eval:
        test_split = test_split.select(range(max_eval))
    
    if "e2a_pruning" in exp_name:
        prune_level_str = Path(base_model_name).name.split("_pruned_")[1].split("_seed")[0]
        payload_dir_name = f"vq_{exp_name}_{cfg.model.pretrained_name.replace('/', '-')}_K{K}_level{prune_level_str}"
    else:
        payload_dir_name = f"vq_{cfg.model.pretrained_name.replace('/', '-')}_K{K}"
    payload_dir = paths.payloads / payload_dir_name
    payload_dir.mkdir(parents=True, exist_ok=True)
    recons_path = payload_dir / "recons.jsonl"

    total_bits, total_tokens, total_chars, n_docs = 0.0, 0, 0, 0
    with recons_path.open("w", encoding="utf-8") as fout:
        for ex in test_split:
            txt = ex.get(text_field) or ""
            if not txt.strip(): continue
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=cfg.data.processing.max_length, add_special_tokens=False)["input_ids"].to(device)
            if ids.numel() == 0 or ids.shape[-1] == 0: continue
            seq_idx = encode_indices(model_vq, ids)[0].tolist()
            if not seq_idx: continue
            bits = cross_entropy_bits_index_stream(lm, seq_idx)
            total_bits += bits
            total_tokens += max(0, len(seq_idx) - 1)
            total_chars += len(txt)
            n_docs += 1
            with torch.no_grad():
                idx_t = torch.tensor(seq_idx, dtype=torch.long, device=device).unsqueeze(0)
                toks_pred = model_vq.decode_from_indices(idx_t)[0].tolist()
            recon = tok.decode(toks_pred, skip_special_tokens=True)
            doc_rec = {"original": txt, "reconstruction": recon, "orig_chars": len(txt), "token_bits": int(round(bits)), "position_bits": 0, "index_len": int(len(seq_idx)), "kept_tokens": 0}
            fout.write(json.dumps(doc_rec) + "\n")

    bpt = (total_bits / max(1, total_tokens)) if total_tokens > 0 else None
    bpc = total_bits / max(1, total_chars) if total_chars > 0 else math.inf
    summary = {"method": "VQ", "model": cfg.model.pretrained_name, "codebook_K": K, "index_bits": float(total_bits), "tokens": int(total_tokens), "chars": int(total_chars), "bpt": (float(bpt) if bpt is not None else None), "bpc": float(bpc), "n_docs": int(n_docs)}
    (payload_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"[stage2_vq] Wrote {recons_path} and summary: bpc={bpc:.6f}, bpt={bpt}")

if __name__ == "__main__":
    main()
