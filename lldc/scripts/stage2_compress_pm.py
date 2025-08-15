# lldc/scripts/stage2_compress_pm.py

from __future__ import annotations
from typing import Any, List
import json, math
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.compression.predictive_masking import pll_surprisal_scores
from lldc.compression.masking_policies import choose_mask
from lldc.compression.payload_codec import arithmetic as ac
from lldc.compression.payload_codec import bitmask as bm
from lldc.compression.payload_codec import rle_elias as rle
from lldc.compression.reconstruction import reconstruct_mlm_text
from lldc.compression.token_alignment import (
    kept_char_spans_from_offsets,
    select_oracle_token_ids_from_spans,
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
    paths = Paths().ensure()
    model_load_path = getattr(cfg, "model_ckpt", cfg.model.pretrained_name)
    log.info(f"[PM Compression] Loading MLM from: {model_load_path}")
    mlm = AutoModelForMaskedLM.from_pretrained(model_load_path)
    tok_mlm = AutoTokenizer.from_pretrained(model_load_path)
    if tok_mlm.pad_token is None and tok_mlm.eos_token:
        tok_mlm.pad_token = tok_mlm.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlm.to(device).eval()
    oracle_name = getattr(
        cfg.experiment.stage2.pm.arithmetic_coder, "oracle_model", "gpt2-large"
    )
    oracle_tok = AutoTokenizer.from_pretrained(oracle_name)
    oracle = AutoModelForCausalLM.from_pretrained(oracle_name).to(device).eval()
    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    text_field = cfg.data.processing.text_field
    split = ds[cfg.data.source.split_map.test]

    out_dir = paths.payloads / f"pm_{Path(str(model_load_path)).name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "recons.jsonl"

    token_bits_total = 0
    pos_bits_total = 0
    chars_total = 0

    with out_file.open("w", encoding="utf-8") as fout:
        for ex in split:
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
            keep_fraction = float(
                getattr(cfg.experiment.stage2.pm, "keep_fraction", 0.4)
            )
            keep_flags = choose_mask("topk_global", s_bits, keep_fraction=keep_fraction)
            _, pos_payload, pos_bits = _position_codec(keep_flags.tolist())
            pos_bits_total += pos_bits
            spans = kept_char_spans_from_offsets(tok_mlm, text, keep_flags)
            kept_oracle_ids = select_oracle_token_ids_from_spans(
                oracle_tok, text, spans
            )
            syms: List[int] = []
            probs: List[List[float]] = []
            if kept_oracle_ids:
                out = oracle(
                    input_ids=torch.tensor(
                        [[oracle_tok.bos_token_id or oracle_tok.eos_token_id or 0]],
                        device=device,
                    )
                )
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                p = torch.softmax(logits, dim=-1)
                for tid in kept_oracle_ids:
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
                    token_bits_total += ac.payload_num_bits(payload)
                except Exception:
                    eps = 1e-12
                    xb = 0.0
                    for sym, pvec in zip(syms, probs):
                        pr = float(pvec[sym]) if 0 <= sym < len(pvec) else 0.0
                        xb += -math.log2(max(pr, eps))
                    token_bits_total += int(round(xb))
            recon_text = reconstruct_mlm_text(tok_mlm, mlm, ids, keep_flags)

            chars_total += len(text)
            fout.write(
                json.dumps(
                    {
                        "original": text,
                        "reconstruction": recon_text,
                        "orig_chars": len(text),
                        "kept_tokens": int(keep_flags.sum().item()),
                        "token_bits": int(token_bits_total),
                        "position_bits": int(pos_bits_total),
                    }
                )
                + "\n"
            )

    bpc = (token_bits_total + pos_bits_total) / max(1, chars_total)
    log.info(
        f"[PM Compression] Done. chars={chars_total}, bits={token_bits_total + pos_bits_total}, bpc={bpc:.4f} → {out_file}"
    )


if __name__ == "__main__":
    main()
