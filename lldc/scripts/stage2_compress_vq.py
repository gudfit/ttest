# lldc/scripts/stage2_compress_vq.py

from __future__ import annotations
from typing import Any, List, Tuple
import json
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.utils.hydra_utils import resolve_auto
from lldc.utils.determinism import set_determinism

from lldc.models.vq.vq_bottleneck import VQBottleneckWrapper
from lldc.models.vq.cache import cached_checkpoint
from lldc.models.vq.vq_trainer import (
    encode_indices,
    train_index_lm,
    cross_entropy_bits_index_stream,
)
from lldc.metrics.fidelity import character_level_fidelity


def _load_vq_model(cfg: Any) -> Tuple[VQBottleneckWrapper, AutoTokenizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_load_path = getattr(cfg, "model_ckpt", cfg.model.pretrained_name)
    base_name_for_cache = cfg.model.pretrained_name
    layer_after = int(cfg.experiment.stage2.vq.bottleneck.layer_after)
    K = int(cfg.experiment.stage2.vq.codebook_sizes[0])
    beta = float(cfg.experiment.stage2.vq.bottleneck.commitment_beta)
    tok = AutoTokenizer.from_pretrained(base_model_load_path)
    if tok.pad_token is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model_load_path)
    ck = cached_checkpoint(base_name_for_cache, K)
    if ck is None:
        raise FileNotFoundError(
            f"No cached VQ checkpoint found for {base_name_for_cache} (K={K}). "
            "Train it in stage2 VQ training step first."
        )

    model = VQBottleneckWrapper(
        base, layer_after=layer_after, codebook_size=K, beta=beta
    )
    sd = torch.load(Path(ck) / "model.pt", map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, tok


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: Any) -> None:
    log = setup_logging()
    cfg = resolve_auto(cfg)
    set_determinism(int(getattr(cfg, "seed", 17)))
    paths = Paths().ensure()
    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    split_map = cfg.data.source.split_map
    text_field = cfg.data.processing.text_field
    train_split = ds[split_map.train]
    test_split = ds[split_map.test]
    model, tok = _load_vq_model(cfg)
    device = next(model.parameters()).device
    max_for_lm = int(
        getattr(cfg.experiment.stage2.vq.index_lm, "train_samples", 2000) or 2000
    )
    all_indices_for_lm: List[List[int]] = []
    for ex in train_split.select(range(min(max_for_lm, len(train_split)))):
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
        seq_idx = encode_indices(model, ids)[0].tolist()
        if seq_idx:
            all_indices_for_lm.append(seq_idx)

    K = int(cfg.experiment.stage2.vq.codebook_sizes[0])
    lm = train_index_lm(
        all_indices_for_lm,
        K=K,
        hidden=int(cfg.experiment.stage2.vq.index_lm.hidden_size),
        layers=int(cfg.experiment.stage2.vq.index_lm.layers),
        epochs=int(cfg.experiment.stage2.vq.index_lm.epochs),
    )
    total_bits = 0.0
    total_chars = 0
    recons_dir = (
        paths.payloads
        / f"vq_{Path(str(getattr(cfg, 'model_ckpt', cfg.model.pretrained_name))).name}_K{K}"
    )
    recons_dir.mkdir(parents=True, exist_ok=True)
    out = recons_dir / "recons.jsonl"
    with out.open("w", encoding="utf-8") as fout:
        for ex in test_split:
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
            idx = encode_indices(model, ids)[0].tolist()
            total_bits += cross_entropy_bits_index_stream(lm, idx)
            total_chars += len(txt)
            with torch.no_grad():
                idx_t = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)
                toks_pred = model.decode_from_indices(idx_t)[0].tolist()
            recon = tok.decode(toks_pred, skip_special_tokens=True)
            cf = character_level_fidelity(txt, recon)
            fout.write(
                json.dumps(
                    {
                        "original": txt,
                        "reconstruction": recon,
                        "orig_chars": len(txt),
                        "index_len": len(idx),
                        "token_bits": None,
                        "position_bits": 0,
                        "char_fidelity": cf,
                    }
                )
                + "\n"
            )
    bpc = total_bits / max(1, total_chars)
    log.info(
        f"[VQ Compression] Done on TEST only. chars={total_chars}, index_bits={total_bits:.2f}, bpc={bpc:.4f} → {out}"
    )


if __name__ == "__main__":
    main()
