# lldc/scripts/run_scalability_exp.py

from __future__ import annotations
from typing import Any, List, Dict, Tuple
import json, math
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.utils.hydra_utils import resolve_auto
from lldc.utils.determinism import set_determinism
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
from lldc.metrics.crumpled_paper import OracleEnsemble, tcm_pcm_from_texts
from lldc.models.vq.vq_trainer import (
    train_vq_joint,
    encode_indices,
    train_index_lm,
    cross_entropy_bits_index_stream,
)


def _position_codec(flags: List[bool]) -> Tuple[str, bytes, int]:
    n = len(flags)
    keep_frac = (sum(1 for f in flags if f) / max(1, n)) if n > 0 else 0.0
    if keep_frac <= 0.25 or keep_frac >= 0.75:
        payload = rle.encode_rle_elias(flags)
        return "rle_elias", payload, len(payload) * 8
    else:
        payload = bm.pack_bitmask(flags)
        return "bitmask", payload, bm.cost_bits(n)


def _files_matching(root: Path, patterns: Tuple[str, ...]) -> set[Path]:
    out: set[Path] = set()
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file():
                out.add(p.resolve())
    return out


def _sum_bytes(files: set[Path]) -> int:
    tot = 0
    for p in files:
        try:
            tot += p.stat().st_size
        except Exception:
            pass
    return tot


def _compute_static_bits(paths: Paths, cfg: Any) -> int:
    MODEL = ("*.pt", "*.bin", "*.safetensors")
    CODEBK = ("*codebook*.*", "*vq*codebook*.*", "*_codebook*.*")
    INDEXLM = ("*index_lm*.*", "*gru*.*")

    ckpt_root = paths.checkpoints
    model_files = _files_matching(ckpt_root, MODEL)
    codebook_files = _files_matching(ckpt_root, CODEBK)
    idxlm_files = _files_matching(ckpt_root, INDEXLM)
    model_only = model_files.difference(codebook_files).difference(idxlm_files)

    total_bits = (
        _sum_bytes(model_only) + _sum_bytes(codebook_files) + _sum_bytes(idxlm_files)
    ) * 8
    override = int(getattr(getattr(cfg, "experiment", {}), "static_bits_override", 0))
    return int(override if (total_bits == 0 and override > 0) else total_bits)


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg: Any) -> None:
    log = setup_logging()
    cfg = resolve_auto(cfg)
    torch.use_deterministic_algorithms(True, warn_only=False)
    set_determinism(int(getattr(cfg, "seed", 17)))
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
    cp_oracle = OracleEnsemble(model_names=[oracle_name], device=device)
    static_bits_total = _compute_static_bits(paths, cfg)

    for size in sizes:
        tr = train.select(range(min(size, len(train))))
        te = test.select(range(min(size, len(test))))
        pm_token_bits = 0
        pm_position_bits = 0
        pm_chars = 0
        pm_fids: List[float] = []
        pm_tcms: List[float] = []
        pm_pcms: List[float] = []
        pm_recons_path = Path("results/subsets") / str(int(size)) / "pm_recons.jsonl"
        pm_recons_path.parent.mkdir(parents=True, exist_ok=True)

        with pm_recons_path.open("w", encoding="utf-8") as fout:
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
                spans = kept_char_spans_from_offsets(tok_mlm, text, keep_flags)
                kept_oracle_ids_list = select_oracle_token_ids_from_spans(
                    oracle_tok, text, spans
                )
                kept_text = (
                    oracle_tok.decode(kept_oracle_ids_list, skip_special_tokens=True)
                    if kept_oracle_ids_list
                    else ""
                )
                syms: List[int] = []
                probs: List[List[float]] = []
                if kept_oracle_ids_list:
                    out = oracle(
                        input_ids=torch.tensor(
                            [[oracle_tok.bos_token_id or oracle_tok.eos_token_id or 0]],
                            device=device,
                        )
                    )
                    past = out.past_key_values
                    logits = out.logits[:, -1, :]
                    p = torch.softmax(logits, dim=-1)

                    for tid in kept_oracle_ids_list:
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
                        for sym, pvec in zip(syms, probs):
                            pr = float(pvec[sym]) if 0 <= sym < len(pvec) else 0.0
                            xb += -math.log2(max(pr, eps))
                        pm_token_bits += int(round(xb))

                recon_text = reconstruct_mlm_text(tok_mlm, mlm, ids, keep_flags)
                pm_chars += len(text)
                cf = character_level_fidelity(text, recon_text)
                tcm_pcm = tcm_pcm_from_texts(text, recon_text, cp_oracle)
                pm_fids.append(cf)
                pm_tcms.append(tcm_pcm["tcm_bits"])
                pm_pcms.append(tcm_pcm["pcm_bits"])

                fout.write(
                    json.dumps(
                        {
                            "original": text,
                            "reconstruction": recon_text,
                            "orig_chars": len(text),
                            "kept_tokens": int(keep_flags.sum().item()),
                            "token_bits": int(pm_token_bits),
                            "position_bits": int(pm_position_bits),
                            "char_fidelity": cf,
                            "tcm_bits": tcm_pcm["tcm_bits"],
                            "pcm_bits": tcm_pcm["pcm_bits"],
                        }
                    )
                    + "\n"
                )

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
            epochs=3,
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
                epochs=int(cfg.experiment.stage2.vq.index_lm.epochs) + 2,
            )
            .to(device)
            .eval()
        )

        vq_bits = 0.0
        vq_chars = 0
        vq_fids: List[float] = []
        vq_tcms: List[float] = []
        vq_pcms: List[float] = []
        vq_recons_path = Path("results/subsets") / str(int(size)) / "vq_recons.jsonl"
        with vq_recons_path.open("w", encoding="utf-8") as fout:
            for ex in te:
                text = ex[text_field]
                ids = vq_tok(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=cfg.data.processing.max_length,
                    add_special_tokens=False,
                )["input_ids"].to(device)
                idx = encode_indices(vq_model, ids)[0].tolist()
                vq_bits += cross_entropy_bits_index_stream(idx_lm, idx)
                vq_chars += len(text)
                with torch.no_grad():
                    idx_t = torch.tensor(
                        idx, dtype=torch.long, device=device
                    ).unsqueeze(0)
                    toks_pred = vq_model.decode_from_indices(idx_t)[0].tolist()
                recon_text = vq_tok.decode(toks_pred, skip_special_tokens=True)

                cf = character_level_fidelity(text, recon_text)
                tcm_pcm = tcm_pcm_from_texts(text, recon_text, cp_oracle)
                vq_fids.append(cf)
                vq_tcms.append(tcm_pcm["tcm_bits"])
                vq_pcms.append(tcm_pcm["pcm_bits"])

                fout.write(
                    json.dumps(
                        {
                            "original": text,
                            "reconstruction": recon_text,
                            "orig_chars": len(text),
                            "index_len": len(idx),
                            "token_bits": None,
                            "position_bits": 0,
                            "char_fidelity": cf,
                            "tcm_bits": tcm_pcm["tcm_bits"],
                            "pcm_bits": tcm_pcm["pcm_bits"],
                        }
                    )
                    + "\n"
                )

        vq_bpc = vq_bits / max(1, vq_chars)
        pm_charF_mean = float(sum(pm_fids) / max(1, len(pm_fids))) if pm_fids else 0.0
        vq_charF_mean = float(sum(vq_fids) / max(1, len(vq_fids))) if vq_fids else 0.0
        pm_tcm_mean = float(sum(pm_tcms) / max(1, len(pm_tcms))) if pm_tcms else 0.0
        pm_pcm_mean = float(sum(pm_pcms) / max(1, len(pm_pcms))) if pm_pcms else 0.0
        vq_tcm_mean = float(sum(vq_tcms) / max(1, len(vq_tcms))) if vq_tcms else 0.0
        vq_pcm_mean = float(sum(vq_pcms) / max(1, len(vq_pcms))) if vq_pcms else 0.0
        pm_amort_bpc = (static_bits_total / max(1, pm_chars)) + pm_bpc
        vq_amort_bpc = (static_bits_total / max(1, vq_chars)) + vq_bpc
        out_dir = Path("results/subsets") / str(int(size))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pm_points.json").write_text(
            json.dumps(
                [
                    {
                        "bpc": pm_bpc,
                        "amortised_bpc": pm_amort_bpc,
                        "charF_mean": pm_charF_mean,
                        "tcm_mean": pm_tcm_mean,
                        "pcm_mean": pm_pcm_mean,
                        "method": "PM",
                    }
                ],
                indent=2,
            )
        )
        (out_dir / "vq_points.json").write_text(
            json.dumps(
                [
                    {
                        "bpc": vq_bpc,
                        "amortised_bpc": vq_amort_bpc,
                        "charF_mean": vq_charF_mean,
                        "tcm_mean": vq_tcm_mean,
                        "pcm_mean": vq_pcm_mean,
                        "method": "VQ",
                    }
                ],
                indent=2,
            )
        )
        log.info(
            f"[scalability] size={size} → "
            f"PM: bpc={pm_bpc:.4f}, amort_bpc={pm_amort_bpc:.4f}, charF={pm_charF_mean:.2f}, TCM={pm_tcm_mean:.2f}, PCM={pm_pcm_mean:.2f} | "
            f"VQ: bpc={vq_bpc:.4f}, amort_bpc={vq_amort_bpc:.4f}, charF={vq_charF_mean:.2f}, TCM={vq_tcm_mean:.2f}, PCM={vq_pcm_mean:.2f}"
        )


if __name__ == "__main__":
    main()
