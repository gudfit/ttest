from __future__ import annotations
from typing import Any, List, Dict
import torch
from datasets import load_dataset
from lldc.models.vq.vq_trainer import (
    train_vq_joint,
    encode_indices,
    train_index_lm,
    cross_entropy_bits_index_stream,
)


def compress_dataset_with_vq(cfg: Any) -> Dict[str, float]:
    """
    Train VQ wrapper (or reuse if checkpointing later), encode indices,
    train GRU index LM, and compute payload bits.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Train VQ joint (short budgeted run)
    model, tok = train_vq_joint(
        base_model_name=cfg.model.pretrained_name,
        dataset_name=cfg.data.source.hf_dataset,
        dataset_config=cfg.data.source.hf_config,
        text_field=cfg.data.processing.text_field,
        max_length=cfg.data.processing.max_length,
        layer_after=cfg.experiment.stage2.vq.bottleneck.layer_after,
        codebook_size=int(cfg.experiment.stage2.vq.codebook_sizes[0]),
        lr=5e-5,
        epochs=2,
        beta=cfg.experiment.stage2.vq.bottleneck.commitment_beta,
    )
    model.eval()

    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    split = ds[cfg.data.source.split_map.test]
    text_field = cfg.data.processing.text_field

    all_indices: List[List[int]] = []
    total_chars = 0
    for ex in split.select(range(min(2000, len(split)))):  # budget guard
        txt = ex[text_field]
        toks = tok(
            txt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.data.processing.max_length,
        )
        ids = toks["input_ids"].to(device)
        idx = encode_indices(model, ids)[0].tolist()
        all_indices.append(idx)
        total_chars += len(txt)

    # Train index LM
    K = int(cfg.experiment.stage2.vq.codebook_sizes[0])
    lm = train_index_lm(
        all_indices,
        K=K,
        hidden=int(cfg.experiment.stage2.vq.index_lm.hidden_size),
        layers=int(cfg.experiment.stage2.vq.index_lm.layers),
        epochs=int(cfg.experiment.stage2.vq.index_lm.epochs),
    )

    # Bits for index sequences
    total_bits = 0.0
    total_tokens = 0
    for seq in all_indices:
        total_bits += cross_entropy_bits_index_stream(lm, seq)
        total_tokens += max(0, len(seq) - 1)
    bpt = total_bits / max(1, total_tokens)
    bpc = total_bits / max(1, total_chars)
    return {
        "index_bits": total_bits,
        "tokens": total_tokens,
        "chars": total_chars,
        "bpt": bpt,
        "bpc": bpc,
    }
