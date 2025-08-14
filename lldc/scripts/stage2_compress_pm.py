# lldc/scripts/stage2_compress_pm.py

from __future__ import annotations
from typing import Any, List, Dict, Tuple
import json, math, time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.utils.hydra_utils import resolve_auto
from lldc.utils import wandb_log
from lldc.utils.determinism import set_determinism

from lldc.metrics.fidelity import (
    character_level_fidelity,
    chrf_score,
    bertscore_f1,
    semantic_span_fidelity,
)
from lldc.metrics.crumpled_paper import Oracle as AROracle, tcm_pcm_from_surprisal

from lldc.compression.predictive_masking import pll_surprisal_scores
from lldc.compression.masking_policies import choose_mask
from lldc.compression.payload_codec import bitmask as bm
from lldc.compression.payload_codec import rle_elias as rle
from lldc.compression.payload_codec import arithmetic as ac


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


def _read_chrf_order(cfg: Any, default: int = 6) -> int:
    v = _cfg_get(cfg, "fidelity.chrf.order", None)
    if isinstance(v, (int, float)):
        return int(v)

    ev = getattr(cfg, "eval", None)
    try:
        if isinstance(ev, (list, tuple)):
            for item in ev:
                name = (
                    item.get("name")
                    if isinstance(item, dict)
                    else getattr(item, "name", None)
                )
                if name == "fidelity":
                    cand = (
                        item.get("chrf", {}).get("order")
                        if isinstance(item, dict)
                        else _cfg_get(item, "chrf.order", None)
                    )
                    if isinstance(cand, (int, float)):
                        return int(cand)
                cand = (
                    item.get("chrf", {}).get("order")
                    if isinstance(item, dict)
                    else _cfg_get(item, "chrf.order", None)
                )
                if isinstance(cand, (int, float)):
                    return int(cand)
    except Exception:
        pass

    return int(default)


def _position_codec(bits_keep: List[bool]) -> tuple[str, bytes, int]:
    n = len(bits_keep)
    keep_frac = sum(bits_keep) / max(1, n)
    if keep_frac <= 0.25 or keep_frac >= 0.75:
        payload = rle.encode_rle_elias(bits_keep)
        return "rle_elias", payload, len(payload) * 8
    else:
        payload = bm.pack_bitmask(bits_keep)
        return "bitmask", payload, bm.cost_bits(n)


@torch.no_grad()
def _reconstruct_mlm_text(
    tok, mlm, input_ids: torch.LongTensor, keep_flags: torch.Tensor
) -> str:
    masked = input_ids.clone()
    mask_id = tok.mask_token_id
    masked[~keep_flags] = mask_id
    outputs = mlm(input_ids=masked.unsqueeze(0))
    logits = outputs.logits[0]
    preds = torch.argmax(logits, dim=-1)
    recon = input_ids.clone()
    recon[~keep_flags] = preds[~keep_flags]
    return tok.decode(recon, skip_special_tokens=True)


def _kept_char_spans_from_offsets(
    text: str, tok, keep_flags: torch.Tensor
) -> List[Tuple[int, int]]:
    enc = tok(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=getattr(tok, "model_max_length", 4096),
    )
    spans: List[Tuple[int, int]] = []
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if i < keep_flags.numel() and keep_flags[i].item() and e > s:
            spans.append((int(s), int(e)))
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _select_oracle_token_ids_from_spans(
    text: str, oracle_tok, spans: List[Tuple[int, int]]
) -> torch.LongTensor:
    if not spans:
        return torch.empty(0, dtype=torch.long)
    enc = oracle_tok(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=getattr(oracle_tok, "model_max_length", 4096),
    )
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    kept: List[int] = []
    si = 0
    for tok_id, (s, e) in zip(ids, offs):
        while si < len(spans) and spans[si][1] <= s:
            si += 1
        if si >= len(spans):
            break
        ks, ke = spans[si]
        if e > s and ke > ks and not (e <= ks or s >= ke):
            kept.append(int(tok_id))
    return torch.tensor(kept, dtype=torch.long)


@torch.no_grad()
def _encode_kept_ids_with_oracle_bits(
    kept_ids: torch.LongTensor,
    oracle_tok,
    oracle_model: AutoModelForCausalLM,
) -> tuple[List[int], List[List[float]], int]:
    device = next(oracle_model.parameters()).device
    vocab_size = int(getattr(oracle_model.config, "vocab_size", len(oracle_tok)))
    if kept_ids.numel() == 0:
        return [], [], vocab_size

    kept_ids = kept_ids.to(device)
    start_id = oracle_tok.bos_token_id or oracle_tok.eos_token_id or 0

    out = oracle_model(input_ids=torch.tensor([[start_id]], device=device))
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)

    symbols: List[int] = []
    probs_list: List[List[float]] = []

    for t in range(kept_ids.numel()):
        probs_list.append(probs.squeeze(0).detach().cpu().tolist())
        sym = int(kept_ids[t].item())
        symbols.append(sym)
        step_out = oracle_model(
            input_ids=kept_ids[t : t + 1].view(1, 1),
            past_key_values=past,
            use_cache=True,
        )
        past = step_out.past_key_values
        logits = step_out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

    return symbols, probs_list, vocab_size


def main():
    import hydra

    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        cfg = resolve_auto(cfg)
        paths = Paths().ensure()
        if cfg.model.arch != "mlm":
            log.warning("PM compression is defined for MLMs. Exiting.")
            return

        set_determinism(getattr(cfg, "seed", 13))

        run = wandb_log.start(
            cfg,
            run_name=f"S2-PM-{cfg.model.name}-seed{getattr(cfg,'seed',13)}",
            tags=["stage2", "pm"],
        )

        model_name = cfg.model.pretrained_name
        tok = AutoTokenizer.from_pretrained(model_name)
        mlm = AutoModelForMaskedLM.from_pretrained(
            getattr(cfg, "model_ckpt", model_name)
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlm.to(device).eval()

        oracle_name = _cfg_get(
            cfg,
            "stage2.pm.arithmetic_coder.oracle_model",
            _cfg_get(
                cfg, "experiment.stage2.pm.arithmetic_coder.oracle_model", "gpt2-large"
            ),
        )
        oracle_tok = AutoTokenizer.from_pretrained(oracle_name)
        if not getattr(oracle_tok, "is_fast", False) or not getattr(
            tok, "is_fast", False
        ):
            raise RuntimeError(
                "Fast tokenizers required for offset mapping alignment (MLM & oracle)."
            )
        oracle = AutoModelForCausalLM.from_pretrained(oracle_name).to(device).eval()
        cp_oracle = AROracle(oracle_name, device=device)

        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        split = ds[cfg.data.source.split_map.test]
        text_field = cfg.data.processing.text_field

        seed = getattr(cfg, "seed", 13)
        payload_dir = paths.payloads / f"pm_{model_name}_seed{seed}"
        payload_dir.mkdir(parents=True, exist_ok=True)
        dump_path = payload_dir / "recons.jsonl"
        if dump_path.exists():
            dump_path.unlink()

        points: List[Dict] = []
        strategies = list(
            _cfg_get(
                cfg,
                "stage2.pm.strategies",
                _cfg_get(cfg, "experiment.stage2.pm.strategies", []),
            )
        )
        mask_rates = list(
            _cfg_get(
                cfg,
                "stage2.pm.mask_rates",
                _cfg_get(cfg, "experiment.stage2.pm.mask_rates", []),
            )
        )
        chrf_order = _read_chrf_order(cfg, default=6)

        all_bpc: List[float] = []
        all_fid: List[float] = []

        for strategy in strategies:
            for rate in mask_rates:
                keep_fraction = 1.0 - float(rate)
                totals = {
                    "position_bits": 0,
                    "token_bits": 0,
                    "chars": 0,
                    "kept_tokens": 0,
                }
                charF_scores: List[float] = []
                chrf_scores: List[float] = []
                berts_scores: List[float] = []
                sem_scores: List[float] = []
                tcm_vals: List[float] = []
                pcm_vals: List[float] = []
                cpu_decode_ms: List[float] = []

                for doc_id, example in enumerate(split):
                    text = example[text_field]
                    toks = tok(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=cfg.data.processing.max_length,
                        add_special_tokens=False,
                    )
                    input_ids = toks["input_ids"][0].to(device)

                    surprisal = pll_surprisal_scores(
                        input_ids, mlm, tok, tok.mask_token_id
                    )
                    keep_flags = choose_mask(strategy, surprisal, keep_fraction).to(
                        device
                    )

                    codec_name, pos_payload, pos_bits = _position_codec(
                        keep_flags.tolist()
                    )

                    kept_spans = _kept_char_spans_from_offsets(text, tok, keep_flags)
                    kept_ids = _select_oracle_token_ids_from_spans(
                        text, oracle_tok, kept_spans
                    )
                    kept_syms, kept_probs, _ = _encode_kept_ids_with_oracle_bits(
                        kept_ids, oracle_tok, oracle
                    )

                    kept_tokens = int(len(kept_syms))

                    token_bits = 0
                    token_decode_ms = 0.0
                    token_payload_status = "none"
                    if kept_syms:
                        try:
                            token_payload = ac.encode_with_probs(kept_syms, kept_probs)
                            token_bits = ac.payload_num_bits(token_payload)
                            t0 = time.perf_counter()
                            _ = ac.decode_with_probs(token_payload, kept_probs)
                            token_decode_ms = (time.perf_counter() - t0) * 1000.0
                            token_payload_status = "encoded_arithmetic"
                        except Exception:
                            eps = 1e-12
                            xb = 0.0
                            for sym, p in zip(kept_syms, kept_probs):
                                pr = float(p[sym]) if 0 <= sym < len(p) else 0.0
                                xb += -math.log2(max(pr, eps))
                            token_bits = int(round(xb))
                            token_decode_ms = 0.0
                            token_payload_status = "estimated_xent"

                    pos_decode_ms = 0.0
                    if codec_name == "bitmask":
                        t0 = time.perf_counter()
                        _ = bm.unpack_bitmask(pos_payload, n_tokens=input_ids.numel())
                        pos_decode_ms = (time.perf_counter() - t0) * 1000.0
                    else:
                        t0 = time.perf_counter()
                        _ = rle.decode_rle_elias(
                            pos_payload, n_tokens=input_ids.numel()
                        )
                        pos_decode_ms = (time.perf_counter() - t0) * 1000.0

                    cpu_decode_ms.append(pos_decode_ms + token_decode_ms)

                    recon_text = _reconstruct_mlm_text(tok, mlm, input_ids, keep_flags)
                    charF = character_level_fidelity(text, recon_text)
                    chrF = chrf_score(text, recon_text, order=int(chrf_order))
                    bF1 = bertscore_f1(text, recon_text, model_type="roberta-large")
                    sspan = semantic_span_fidelity(text, recon_text)

                    charF_scores.append(charF)
                    chrf_scores.append(chrF)
                    berts_scores.append(bF1)
                    sem_scores.append(sspan)

                    if doc_id % 10 == 0:
                        s_orig = cp_oracle.surprisal_bits(text)[1]
                        s_rec = cp_oracle.surprisal_bits(recon_text)[1]
                        m = tcm_pcm_from_surprisal(
                            s_orig, s_rec, vocab_size=len(oracle_tok)
                        )
                        tcm_vals.append(float(m["tcm_bits"]))
                        pcm_vals.append(float(m["pcm_bits"]))

                    bpt_doc = float(token_bits) / max(1, kept_tokens)

                    with open(dump_path, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "doc_id": doc_id,
                                    "strategy": strategy,
                                    "mask_rate": rate,
                                    "position_codec": codec_name,
                                    "position_bits": pos_bits,
                                    "token_bits": token_bits,
                                    "token_payload_status": token_payload_status,
                                    "kept_tokens": kept_tokens,
                                    "bpt": bpt_doc,
                                    "orig_chars": len(text),
                                    "original": text,
                                    "reconstruction": recon_text,
                                    "char_fidelity": charF,
                                    "chrf": chrF,
                                    "bertscore_f1": bF1,
                                    "semantic_span_fid": sspan,
                                    "cpu_decode_ms": pos_decode_ms + token_decode_ms,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    totals["position_bits"] += pos_bits
                    totals["token_bits"] += token_bits
                    totals["chars"] += len(text)
                    totals["kept_tokens"] += kept_tokens

                bpc = (totals["position_bits"] + totals["token_bits"]) / max(
                    1, totals["chars"]
                )
                bpt_point = float(totals["token_bits"]) / max(1, totals["kept_tokens"])

                char_fid_point = float(sum(charF_scores) / max(1, len(charF_scores)))
                chrf_point = float(sum(chrf_scores) / max(1, len(chrf_scores)))
                berts_point = float(sum(berts_scores) / max(1, len(berts_scores)))
                sem_point = float(sum(sem_scores) / max(1, len(sem_scores)))
                tcm_point = (
                    float(sum(tcm_vals) / max(1, len(tcm_vals))) if tcm_vals else 0.0
                )
                pcm_point = (
                    float(sum(pcm_vals) / max(1, len(pcm_vals))) if pcm_vals else 0.0
                )
                cpu_decode_mean = float(sum(cpu_decode_ms) / max(1, len(cpu_decode_ms)))

                all_bpc.append(bpc)
                all_fid.append(char_fid_point)

                point = {
                    "method": "PM",
                    "strategy": strategy,
                    "mask_rate": rate,
                    "bpc": bpc,
                    "bpt": bpt_point,
                    "charF_mean": char_fid_point,
                    "chrf_mean": chrf_point,
                    "bertscore_f1_mean": berts_point,
                    "sem_span_fid_mean": sem_point,
                    "tcm_mean": tcm_point,
                    "pcm_mean": pcm_point,
                    "cpu_decode_ms_mean": cpu_decode_mean,
                }
                points.append(point)
                (paths.rd_curves / "pm_points.json").write_text(
                    json.dumps(points, indent=2)
                )

                wandb_log.log(
                    {
                        "pm/payload_bits": int(totals["token_bits"]),
                        "pm/pos_bits": int(totals["position_bits"]),
                        "pm/bpc": float(bpc),
                        "pm/bpt": float(bpt_point),
                        "pm/char_fid": float(char_fid_point),
                        "pm/chrf": float(chrf_point),
                        "pm/bertscore_f1": float(berts_point),
                        "pm/sem_span_fid": float(sem_point),
                        "pm/tcm": float(tcm_point),
                        "pm/pcm": float(pcm_point),
                        "pm/cpu_decode_ms_mean": float(cpu_decode_mean),
                    }
                )

        bpc_mean = sum(all_bpc) / max(1, len(all_bpc))
        fid_mean = sum(all_fid) / max(1, len(all_fid))

        wandb_log.log(
            {
                "pm/aggregate_bpc": float(bpc_mean),
                "pm/aggregate_charF": float(fid_mean),
            }
        )
        wandb_log.finish()

    _run()


if __name__ == "__main__":
    main()
