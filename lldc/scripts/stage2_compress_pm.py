# lldc/scripts/stage2_compress_pm.py
from __future__ import annotations
from typing import Any, List, Tuple
import json
import math
import time
from pathlib import Path
import hydra
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from lldc.utils.paths import Paths
from lldc.utils.logging import setup_logging
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
from lldc.compressors.pm.token_coder import encode_kept_stream_with_oracle
from lldc.metrics.fidelity import character_level_fidelity
def _position_codec(flags: List[bool]) -> Tuple[str, bytes, int]:
    n = len(flags)
    keep_frac = (sum(1 for f in flags if f) / max(1, n)) if n > 0 else 0.0
    if keep_frac <= 0.25 or keep_frac >= 0.75:
        payload = rle.encode_rle_elias(flags)
        return "rle_elias", payload, len(payload) * 8
    else:
        payload = bm.pack_bitmask(flags)
        return "bitmask", payload, bm.cost_bits(n)
def _reconstruct_beam(
    tok, mlm, input_ids: torch.LongTensor, keep_flags: torch.Tensor, beam_width: int = 4
) -> str:
    device = next(mlm.parameters()).device
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise RuntimeError("Tokenizer must define a [MASK] token.")
    ids = input_ids.clone().to(device)
    masked = ids.clone()
    masked[~keep_flags] = int(mask_id)
    positions = torch.nonzero(~keep_flags, as_tuple=False).flatten().tolist()
    if not positions:
        return tok.decode(ids, skip_special_tokens=True)
    bad_ids = set(getattr(tok, "all_special_ids", []) or [])
    bad_ids.add(int(mask_id))
    bad_ids = torch.tensor(sorted(bad_ids), device=device, dtype=torch.long)
    vocab_size = getattr(mlm.config, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = mlm.get_output_embeddings().num_embeddings
        except Exception:
            vocab_size = len(tok)
    banned_bias = torch.zeros(int(vocab_size), device=device)
    if bad_ids.numel() > 0:
        banned_bias[bad_ids] = float("-inf")
    def make_attn(seq_2d: torch.LongTensor) -> torch.LongTensor:
        pad_id = tok.pad_token_id
        if pad_id is None:
            return torch.ones_like(seq_2d, dtype=torch.long)
        return (seq_2d != pad_id).long()
    beams: List[Tuple[torch.LongTensor, float]] = [(masked.unsqueeze(0), 0.0)]
    with torch.inference_mode():
        for pos in positions:
            batch = torch.cat([seq for (seq, _) in beams], dim=0)
            attn = make_attn(batch)
            logits = mlm(input_ids=batch, attention_mask=attn).logits[:, pos, :]
            logp = F.log_softmax(logits, dim=-1) + banned_bias
            candidates = {}
            for i, (seq, base_score) in enumerate(beams):
                vals, idxs = torch.topk(logp[i], k=min(beam_width, logp.size(-1)))
                for lp, tid in zip(vals.tolist(), idxs.tolist()):
                    if lp == float("-inf"):
                        continue
                    s2 = seq.clone()
                    s2[0, pos] = int(tid)
                    key = s2[0].detach().cpu().numpy().tobytes()
                    sc = base_score + float(lp)
                    if (key not in candidates) or (sc > candidates[key][1]):
                        candidates[key] = (s2, sc)
            beams = sorted(candidates.values(), key=lambda x: x[1], reverse=True)[
                :beam_width
            ]
        if len(beams) > 1:
            final_batch = torch.cat([seq for (seq, _) in beams], dim=0)
            pll_scores = torch.zeros(final_batch.size(0), device=device)
            for pos in positions:
                tmp = final_batch.clone()
                true_tok = tmp[:, pos].clone()
                tmp[:, pos] = int(mask_id)
                attn = make_attn(tmp)
                logits = mlm(input_ids=tmp, attention_mask=attn).logits[:, pos, :]
                logp = F.log_softmax(logits, dim=-1)
                pll_scores += logp[torch.arange(tmp.size(0), device=device), true_tok]
            best_idx = int(torch.argmax(pll_scores).item())
            best_ids = beams[best_idx][0][0]
        else:
            best_ids = beams[0][0][0]
    return tok.decode(best_ids, skip_special_tokens=True)
def _reconstruct_nucleus(
    tok, mlm, input_ids: torch.LongTensor, keep_flags: torch.Tensor, top_p: float = 0.9
) -> str:
    device = next(mlm.parameters()).device
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise RuntimeError("Tokenizer must define a [MASK] token.")
    ids = input_ids.clone().to(device)
    masked = ids.clone()
    masked[~keep_flags] = int(mask_id)
    positions = torch.nonzero(~keep_flags, as_tuple=False).flatten().tolist()
    if not positions:
        return tok.decode(ids, skip_special_tokens=True)
    seq = masked.unsqueeze(0)
    for pos in positions:
        out = mlm(input_ids=seq).logits[0, pos]
        probs = torch.softmax(out, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = (cumsum > top_p).nonzero(as_tuple=False)
        K = int(cutoff[0, 0].item()) + 1 if cutoff.numel() > 0 else probs.numel()
        p = sorted_probs[:K] / max(1e-12, float(sorted_probs[:K].sum().item()))
        choice = torch.multinomial(p, num_samples=1).item()
        tid = int(sorted_idx[choice].item())
        seq[0, pos] = tid
    return tok.decode(seq[0], skip_special_tokens=True)
@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: Any) -> None:
    log = setup_logging()
    paths = Paths().ensure()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlm_name = cfg.model.pretrained_name
    tok_mlm = AutoTokenizer.from_pretrained(mlm_name)
    if tok_mlm.pad_token is None and tok_mlm.eos_token:
        tok_mlm.pad_token = tok_mlm.eos_token
    mlm = AutoModelForMaskedLM.from_pretrained(mlm_name).to(device).eval()
    oracle_name = getattr(
        getattr(cfg.experiment.stage2, "pm", None), "arithmetic_coder", {}
    ).get("oracle_model", "gpt2-large")
    oracle_tok = AutoTokenizer.from_pretrained(oracle_name)
    if oracle_tok.pad_token is None and oracle_tok.eos_token:
        oracle_tok.pad_token = oracle_tok.eos_token
    oracle_model = AutoModelForCausalLM.from_pretrained(oracle_name).to(device).eval()
    ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
    split_map = cfg.data.source.split_map
    text_field = cfg.data.processing.text_field
    test_split = ds[split_map.test].select(range(1000))
    max_eval = int(
        getattr(getattr(cfg, "data", {}), "limits", {}).get("max_eval_samples", 2000)
        or 2000
    )
    keep_fraction = float(
        getattr(getattr(cfg.experiment.stage2, "pm", None), "keep_fraction", 0.4)
    )
    policy = str(
        getattr(getattr(cfg.experiment.stage2, "pm", None), "policy", "topk_global")
    )
    dec_cfg = getattr(getattr(cfg.experiment.stage2, "pm", None), "decoding", {})
    dec_method = str(getattr(dec_cfg, "method", "greedy")).lower()
    beam_width = int(getattr(dec_cfg, "beam_width", 4))
    top_p = float(getattr(dec_cfg, "top_p", 0.9))
    tag_mask = f"mask{keep_fraction:.2f}"
    payload_dir = paths.payloads / f"pm_{mlm_name.replace('/', '-')}_{tag_mask}"
    payload_dir.mkdir(parents=True, exist_ok=True)
    recons_path = payload_dir / "recons.jsonl"
    n_docs = 0
    total_token_bits = 0
    total_position_bits = 0
    total_chars = 0
    with recons_path.open("w", encoding="utf-8") as fout:
        for ex in test_split.select(range(min(max_eval, len(test_split)))):
            text = (ex.get(text_field) or "").strip()
            if not text:
                continue
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
            keep_flags_t = choose_mask(policy, s_bits, keep_fraction=keep_fraction)
            keep_flags = keep_flags_t.tolist()
            pos_codec, pos_payload, pos_bits = _position_codec(keep_flags)
            total_position_bits += pos_bits
            spans = kept_char_spans_from_offsets(tok_mlm, text, keep_flags)
            kept_oracle_ids = select_oracle_token_ids_from_spans(
                oracle_tok, text, spans
            )
            token_bits = 0
            kept_tokens = len(kept_oracle_ids)
            if kept_oracle_ids:
                syms, probs_list, _ = encode_kept_stream_with_oracle(
                    kept_text=oracle_tok.decode(
                        kept_oracle_ids, skip_special_tokens=True
                    ),
                    oracle_tok=oracle_tok,
                    oracle_model=oracle_model,
                )
                try:
                    token_payload = ac.encode_with_probs(syms, probs_list)
                    token_bits = ac.payload_num_bits(token_payload)
                except Exception:
                    eps = 1e-12
                    xb = 0.0
                    for sym, pvec in zip(syms, probs_list):
                        pr = float(pvec[sym]) if 0 <= sym < len(pvec) else 0.0
                        xb += -math.log2(max(pr, eps))
                    token_bits = int(round(xb))
            total_token_bits += token_bits
            t0 = time.perf_counter()
            if pos_codec == "rle_elias":
                _ = rle.decode_rle_elias(pos_payload, ids.numel())
            else:
                _ = bm.unpack_bitmask(pos_payload, ids.numel())
            if kept_oracle_ids:
                try:
                    _ = ac.decode_with_probs(token_payload, probs_list)
                except Exception:
                    pass
            t_payload_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            if dec_method == "greedy":
                recon_text = reconstruct_mlm_text(tok_mlm, mlm, ids, keep_flags_t)
            elif dec_method == "beam":
                recon_text = _reconstruct_beam(
                    tok_mlm, mlm, ids, keep_flags_t, beam_width=beam_width
                )
            elif dec_method == "nucleus":
                recon_text = _reconstruct_nucleus(
                    tok_mlm, mlm, ids, keep_flags_t, top_p=top_p
                )
            else:
                recon_text = reconstruct_mlm_text(tok_mlm, mlm, ids, keep_flags_t)
            t_mlm_ms = (time.perf_counter() - t1) * 1000.0
            cpu_decode_ms = t_payload_ms + t_mlm_ms
            charF = character_level_fidelity(text, recon_text)
            doc = {
                "original": text,
                "reconstruction": recon_text,
                "orig_chars": len(text),
                "kept_tokens": int(kept_tokens),
                "token_bits": int(token_bits),
                "position_bits": int(pos_bits),
                "cpu_decode_ms": float(cpu_decode_ms),
                "char_fidelity": float(charF),
                "decoding_method": dec_method,
            }
            fout.write(json.dumps(doc) + "\n")
            total_chars += len(text)
            n_docs += 1
    summary = {
        "method": "PM",
        "model": mlm_name,
        "mask_rate": keep_fraction,
        "n_docs": int(n_docs),
        "total_token_bits": int(total_token_bits),
        "total_position_bits": int(total_position_bits),
        "total_chars": int(total_chars),
        "bpc": (total_token_bits + total_position_bits) / max(1, total_chars),
        "notes": "cpu_decode_ms includes payload decode + MLM forward time; decoding strategies supported (greedy/beam/nucleus).",
    }
    (payload_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        f"[stage2_pm] Wrote {recons_path} | bpc={summary['bpc']:.6f} | docs={n_docs} | decode(ms)=payload+mlm"
    )
if __name__ == "__main__":
    main()
