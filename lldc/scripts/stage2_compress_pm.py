from __future__ import annotations
from typing import Any, List, Dict
import json, math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.utils.hydra_utils import resolve_auto
from lldc.utils import wandb_log

from lldc.metrics.fidelity import character_level_fidelity
from lldc.metrics.crumpled_paper import Oracle as AROracle, tcm_pcm_from_surprisal

from lldc.compression.predictive_masking import pll_surprisal_scores
from lldc.compression.masking_policies import choose_mask
from lldc.compression.payload_codec import bitmask as bm
from lldc.compression.payload_codec import rle_elias as rle
from lldc.compression.payload_codec import arithmetic as ac


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
    logits = outputs.logits[0]  # [T,V]
    preds = torch.argmax(logits, dim=-1)
    recon = input_ids.clone()
    recon[~keep_flags] = preds[~keep_flags]
    return tok.decode(recon, skip_special_tokens=True)


def _kept_substring_from_offsets(text: str, tok, keep_flags: torch.Tensor) -> str:
    enc = tok(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=getattr(tok, "model_max_length", 4096),
    )
    kept_chunks: List[str] = []
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if i < keep_flags.numel() and keep_flags[i].item() and e > s:
            kept_chunks.append(text[s:e])
    return "".join(kept_chunks)


@torch.no_grad()
def _encode_kept_with_oracle_bits(
    kept_text: str,
    oracle_tok,
    oracle_model: AutoModelForCausalLM,
) -> tuple[List[int], List[List[float]], int]:
    device = next(oracle_model.parameters()).device
    vocab_size = int(getattr(oracle_model.config, "vocab_size", len(oracle_tok)))

    if not kept_text:
        return [], [], vocab_size

    enc = oracle_tok(kept_text, return_tensors="pt", add_special_tokens=False)
    kept_ids = enc["input_ids"][0].to(device)  # [Lk]

    # seed with start token
    start_id = oracle_tok.bos_token_id or oracle_tok.eos_token_id
    if start_id is None:
        start_id = 0

    # First step: run start token to build initial cache
    out = oracle_model(input_ids=torch.tensor([[start_id]], device=device))
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)  # [1,V]

    symbols: List[int] = []
    probs_list: List[List[float]] = []

    for t in range(kept_ids.numel()):
        # Distribution for current symbol t (conditioned on previous kepts)
        probs_list.append(probs.squeeze(0).detach().cpu().tolist())
        sym = int(kept_ids[t].item())
        symbols.append(sym)

        # Teacher-force true symbol into cache and advance
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

        run = wandb_log.start(
            cfg, run_name=f"S2-PM-{cfg.model.name}", tags=["stage2", "pm"]
        )

        # Load models
        model_name = cfg.model.pretrained_name
        tok = AutoTokenizer.from_pretrained(model_name)
        mlm = AutoModelForMaskedLM.from_pretrained(
            getattr(cfg, "model_ckpt", model_name)
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlm.to(device).eval()

        # AR oracle for token payload + Crumpled Paper measurement tokenizer
        oracle_name = (
            cfg.experiment.stage2.pm.arithmetic_coder.oracle_model
            if "experiment" in cfg and "stage2" in cfg.experiment
            else "gpt2-large"
        )
        oracle_tok = AutoTokenizer.from_pretrained(oracle_name)
        oracle = AutoModelForCausalLM.from_pretrained(oracle_name).to(device).eval()
        cp_oracle = AROracle(oracle_name, device=device)

        # Data
        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        split = ds[cfg.data.source.split_map.test]
        text_field = cfg.data.processing.text_field

        payload_dir = paths.payloads / f"pm_{model_name}"
        payload_dir.mkdir(parents=True, exist_ok=True)
        dump_path = payload_dir / "recons.jsonl"
        if dump_path.exists():
            dump_path.unlink()

        points: List[Dict] = []
        strategies = list(cfg.experiment.stage2.pm.strategies)
        mask_rates = list(cfg.experiment.stage2.pm.mask_rates)

        all_bpc: List[float] = []
        all_fid: List[float] = []

        for strategy in strategies:
            for rate in mask_rates:
                keep_fraction = 1.0 - float(rate)
                totals = {"position_bits": 0, "token_bits": 0, "chars": 0}
                charF_scores: List[float] = []
                tcm_vals: List[float] = []
                pcm_vals: List[float] = []

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
                    keep_flags = choose_mask(strategy, surprisal, keep_fraction)
                    keep_flags = keep_flags.to(device)

                    codec_name, pos_payload, pos_bits = _position_codec(
                        keep_flags.tolist()
                    )

                    kept_text = _kept_substring_from_offsets(text, tok, keep_flags)
                    kept_syms, kept_probs, _ = _encode_kept_with_oracle_bits(
                        kept_text, oracle_tok, oracle
                    )
                    if kept_syms:
                        token_payload = ac.encode_with_probs(kept_syms, kept_probs)
                        token_bits = ac.payload_num_bits(token_payload)
                    else:
                        token_bits = 0

                    recon_text = _reconstruct_mlm_text(tok, mlm, input_ids, keep_flags)
                    charF = character_level_fidelity(text, recon_text)
                    charF_scores.append(charF)

                    if doc_id % 10 == 0:
                        s_orig = cp_oracle.surprisal_bits(text)
                        s_rec = cp_oracle.surprisal_bits(recon_text)
                        m = tcm_pcm_from_surprisal(
                            s_orig, s_rec, vocab_size=len(oracle_tok)
                        )
                        tcm_vals.append(float(m["tcm_bits"]))
                        pcm_vals.append(float(m["pcm_bits"]))

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
                                    "original": text,
                                    "reconstruction": recon_text,
                                }
                            )
                            + "\n"
                        )

                    totals["position_bits"] += pos_bits
                    totals["token_bits"] += token_bits
                    totals["chars"] += len(text)

                bpc = (totals["position_bits"] + totals["token_bits"]) / max(
                    1, totals["chars"]
                )

                char_fid_point = float(sum(charF_scores) / max(1, len(charF_scores)))
                tcm_point = (
                    float(sum(tcm_vals) / max(1, len(tcm_vals))) if tcm_vals else 0.0
                )
                pcm_point = (
                    float(sum(pcm_vals) / max(1, len(pcm_vals))) if pcm_vals else 0.0
                )

                all_bpc.append(bpc)
                all_fid.append(char_fid_point)

                point = {
                    "method": "PM",
                    "strategy": strategy,
                    "mask_rate": rate,
                    "bpc": bpc,
                    "charF_mean": char_fid_point,
                    "tcm_mean": tcm_point,
                    "pcm_mean": pcm_point,
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
                        "pm/char_fid": float(char_fid_point),
                        "pm/tcm": float(tcm_point),
                        "pm/pcm": float(pcm_point),
                    }
                )

        bpc_mean = sum(all_bpc) / max(1, len(all_bpc))
        fid_mean = sum(all_fid) / max(1, len(all_fid))

        wandb_log.log(
            {
                "pm/aggregate_bpc": float(bpc_mean),
                "pm/aggregate_fid": float(fid_mean),
            }
        )
        wandb_log.finish()

    _run()


if __name__ == "__main__":
    main()
