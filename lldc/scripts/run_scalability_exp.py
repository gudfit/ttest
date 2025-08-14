# lldc/scripts/run_scalability_exp.py

from __future__ import annotations
from typing import Any, List, Dict
import json
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.utils.hydra_utils import resolve_auto
from lldc.compression.predictive_masking import pll_surprisal_scores
from lldc.compression.masking_policies import choose_mask
from lldc.compression.payload_codec import arithmetic as ac
from lldc.compression.payload_codec import bitmask as bm
from lldc.compression.payload_codec import rle_elias as rle

from lldc.models.vq.vq_trainer import (
    train_vq_joint,
    encode_indices,
    train_index_lm,
    cross_entropy_bits_index_stream,
)


def _position_codec(bits_keep):
    n = len(bits_keep)
    keep_frac = sum(bits_keep) / max(1, n)
    if keep_frac <= 0.25 or keep_frac >= 0.75:
        payload = rle.encode_rle_elias(bits_keep)
        return "rle_elias", payload, len(payload) * 8
    else:
        payload = bm.pack_bitmask(bits_keep)
        return "bitmask", payload, bm.cost_bits(n)


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: Any) -> None:
    log = setup_logging()
    cfg = resolve_auto(cfg)
    paths = Paths().ensure()
    dataset_name = getattr(cfg.e2b.scalability, "dataset", "wikitext-2")
    factors = list(getattr(cfg.e2b.scalability, "factors", [1, 2, 4, 8, 16, 32]))
    schedule_default = [100, 500, 1000, 5000, 10000, 20000]
    if dataset_name == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        text_field = "text"
    else:
        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        text_field = cfg.data.processing.text_field

    train = ds[cfg.data.source.split_map.train]
    test = ds[cfg.data.source.split_map.test]
    total_train = len(train)
    sizes: List[int] = []
    for f in factors or []:
        n = max(10, int(total_train * (float(f) / float(max(factors)))))
        sizes.append(n)
    if not sizes:
        sizes = [s for s in schedule_default if s <= total_train]

    sizes = sorted(set(max(10, min(s, total_train)) for s in sizes))
    log.info(f"[scalability] Running sizes={sizes}")
    mlm_name = cfg.model.pretrained_name
    mlm = AutoModelForMaskedLM.from_pretrained(mlm_name)
    tok_mlm = AutoTokenizer.from_pretrained(mlm_name)
    if tok_mlm.pad_token is None and tok_mlm.eos_token:
        tok_mlm.pad_token = tok_mlm.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlm.to(device).eval()
    oracle_name = getattr(
        cfg.experiment.stage2.pm.arithmetic_coder, "oracle_model", "gpt2-large"
    )
    oracle_tok = AutoTokenizer.from_pretrained(oracle_name)
    oracle = AutoModelForCausalLM.from_pretrained(oracle_name).to(device).eval()

    for size in sizes:
        tr = train.select(range(min(size, len(train))))
        te = test.select(range(min(size, len(test))))
        pm_token_bits = 0
        pm_position_bits = 0
        pm_chars = 0

        for ex in te:
            text = ex[text_field]
            ids = tok_mlm(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.data.processing.max_length,
                add_special_tokens=False,
            )["input_ids"][0].to(device)

            if ids.numel() == 0:
                continue

            s_bits = pll_surprisal_scores(ids, mlm, tok_mlm, tok_mlm.mask_token_id)
            keep_flags = choose_mask("topk_global", s_bits, keep_fraction=0.4)
            codec, pos_payload, pos_bits = _position_codec(keep_flags.tolist())
            pm_position_bits += pos_bits
            kept_mlmtoks = ids[keep_flags]
            if kept_mlmtoks.numel():
                kept_text = tok_mlm.decode(kept_mlmtoks, skip_special_tokens=True)
                kept_ids_oracle = oracle_tok(
                    kept_text, add_special_tokens=False, return_tensors="pt"
                )["input_ids"][0].to(device)
                syms: List[int] = []
                probs: List[List[float]] = []
                if kept_ids_oracle.numel():
                    out = oracle(
                        input_ids=torch.tensor(
                            [[oracle_tok.bos_token_id or 0]], device=device
                        )
                    )
                    past = out.past_key_values
                    logits = out.logits[:, -1, :]
                    p = torch.softmax(logits, dim=-1)
                    for tid in kept_ids_oracle.tolist():
                        probs.append(p.squeeze(0).detach().cpu().tolist())
                        step = oracle(
                            input_ids=torch.tensor([[tid]], device=device),
                            past_key_values=past,
                            use_cache=True,
                        )
                        past = step.past_key_values
                        logits = step.logits[:, -1, :]
                        p = torch.softmax(logits, dim=-1)
                        syms.append(int(tid))
                if syms:
                    try:
                        payload = ac.encode_with_probs(syms, probs)
                        pm_token_bits += ac.payload_num_bits(payload)
                    except Exception:
                        eps = 1e-12
                        xb = 0.0
                        for sym, p in zip(syms, probs):
                            pr = float(p[sym]) if 0 <= sym < len(p) else 0.0
                            xb += -math.log2(max(pr, eps))
                        pm_token_bits += int(round(xb))

            pm_chars += len(text)

        pm_bpc = (pm_token_bits + pm_position_bits) / max(1, pm_chars)
        vq_model, vq_tok = train_vq_joint(
            base_model_name=cfg.model.pretrained_name,
            dataset_name=cfg.data.source.hf_dataset,
            dataset_config=cfg.data.source.hf_config,
            text_field=text_field,
            max_length=cfg.data.processing.max_length,
            layer_after=int(cfg.experiment.stage2.vq.bottleneck.layer_after),
            codebook_size=int(cfg.experiment.stage2.vq.codebook_sizes[0]),
            lr=5e-5,
            epochs=1,
            beta=float(cfg.experiment.stage2.vq.bottleneck.commitment_beta),
        )
        vq_model.eval().to(device)

        K = int(cfg.experiment.stage2.vq.codebook_sizes[0])
        idx_train: List[List[int]] = []
        for ex in tr:
            ids = vq_tok(
                ex[text_field],
                return_tensors="pt",
                truncation=True,
                max_length=cfg.data.processing.max_length,
                add_special_tokens=False,
            )["input_ids"].to(device)
            idx = encode_indices(vq_model, ids)[0].tolist()
            if idx:
                idx_train.append(idx)
        idx_lm = (
            train_index_lm(
                idx_train,
                K=K,
                hidden=int(cfg.experiment.stage2.vq.index_lm.hidden_size),
                layers=int(cfg.experiment.stage2.vq.index_lm.layers),
                epochs=1,
            )
            .to(device)
            .eval()
        )

        vq_bits = 0.0
        vq_chars = 0
        for ex in te:
            ids = vq_tok(
                ex[text_field],
                return_tensors="pt",
                truncation=True,
                max_length=cfg.data.processing.max_length,
                add_special_tokens=False,
            )["input_ids"].to(device)
            idx = encode_indices(vq_model, ids)[0].tolist()
            vq_bits += cross_entropy_bits_index_stream(idx_lm, idx)
            vq_chars += len(ex[text_field])
        vq_bpc = vq_bits / max(1, vq_chars)
        out_dir = Path("results/subsets") / str(int(size))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pm_points.json").write_text(
            json.dumps([{"bpc": pm_bpc, "method": "PM"}], indent=2)
        )
        (out_dir / "vq_points.json").write_text(
            json.dumps([{"bpc": vq_bpc, "method": "VQ"}], indent=2)
        )
        log.info(
            f"[scalability] size={size} → PM bpc={pm_bpc:.4f}, VQ bpc={vq_bpc:.4f}"
        )

    log.info("[scalability] Done.")


if __name__ == "__main__":
    main()
