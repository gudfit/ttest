# lldc/scripts/stage2_compress_vq.py

from __future__ import annotations
from typing import Any, List, Dict, Tuple
import json
from pathlib import Path
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

try:
    import constriction
except Exception:
    constriction = None

from lldc.utils.logging import setup_logging
from lldc.utils.hydra_utils import resolve_auto
from lldc.utils.paths import Paths
from lldc.utils import wandb_log
from lldc.metrics.fidelity import (
    character_level_fidelity,
    chrf_score,
    bertscore_f1,
    semantic_span_fidelity,
)
from lldc.metrics.crumpled_paper import Oracle as AROracle, tcm_pcm_from_surprisal
from lldc.models.vq.vq_trainer import (
    train_vq_joint,
    encode_indices,
    train_index_lm,
    cross_entropy_bits_index_stream,
    VQBottleneckWrapper,
    save_vq_wrapper,
    load_vq_wrapper,
)
from lldc.decompression.vq_decompress import reconstruct_tokens_from_indices
from lldc.models.vq.cache import cache_dir_for, cached_checkpoint


def _indexlm_step_probs_cpu(lm, prev_sym: int, h=None) -> Tuple[np.ndarray, Any]:
    device = torch.device("cpu")
    with torch.no_grad():
        x = torch.tensor([[prev_sym]], dtype=torch.long, device=device)
        emb = lm.embed(x)
        y, h_next = lm.gru(emb, h)
        logits = lm.out(y[:, -1, :])
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    s = float(probs.sum())
    if s <= 0.0 or not np.isfinite(s):
        probs = np.ones_like(probs) / probs.size
    else:
        probs = probs / s
    return probs, h_next


def _encode_indices_arithmetic_cpu(lm_cpu, indices: List[int]) -> bytes:
    if constriction is None:
        raise RuntimeError("constriction>=0.3 is required for arithmetic payloads.")

    if len(indices) < 2:
        return b""

    coder = constriction.stream.queue_bit_coder.QueueBitCoder()
    prev = int(indices[0])
    h = None
    for sym in indices[1:]:
        probs, h = _indexlm_step_probs_cpu(lm_cpu, prev, h)
        cdf = np.cumsum(probs)
        coder.encode_symbol_using_cdf(int(sym), cdf)
        prev = int(sym)
    return coder.get_compressed()


def _decode_indices_arithmetic_cpu(
    lm_cpu, payload: bytes, first_sym: int, length: int
) -> List[int]:
    if constriction is None:
        raise RuntimeError("constriction>=0.3 is required for arithmetic payloads.")

    if length <= 0:
        return []

    decoder = constriction.stream.queue_bit_coder.QueueBitCoder(payload)
    out: List[int] = []
    prev = int(first_sym)
    h = None
    for _ in range(length):
        probs, h = _indexlm_step_probs_cpu(lm_cpu, prev, h)
        cdf = np.cumsum(probs)
        sym = int(decoder.decode_symbol_using_cdf(cdf))
        out.append(sym)
        prev = sym
    return out


def main():
    import hydra

    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        cfg = resolve_auto(cfg)
        paths = Paths().ensure()
        if cfg.model.arch != "ar":
            log.warning("VQ pipeline is defined for AR models (GPT-2 family). Exiting.")
            return

        run = wandb_log.start(
            cfg, run_name=f"S2-VQ-{cfg.model.name}", tags=["stage2", "vq"]
        )

        base_model_name = getattr(cfg, "model_ckpt", cfg.model.pretrained_name)
        tok = AutoTokenizer.from_pretrained(cfg.model.pretrained_name)
        if tok.pad_token is None and tok.eos_token:
            tok.pad_token = tok.eos_token

        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        split = ds[cfg.data.source.split_map.test]
        text_field = cfg.data.processing.text_field
        cp_oracle = AROracle("gpt2-large")

        Ks = list(cfg.experiment.stage2.vq.codebook_sizes)
        points: List[Dict] = []
        rd_file = paths.rd_curves / "vq_points.json"
        payload_root = paths.payloads

        vq_cfg = getattr(cfg.experiment.stage2, "vq", {})
        reuse_cache = bool(getattr(vq_cfg, "reuse_cache", True))
        force_retrain = bool(getattr(vq_cfg, "force_retrain", False))

        for K in Ks:
            model: VQBottleneckWrapper
            ck_dir = (
                cached_checkpoint(cfg.model.name, int(K))
                if (reuse_cache and not force_retrain)
                else None
            )
            if ck_dir is not None:
                log.info(f"[VQ] Reusing cached VQ wrapper for K={K}: {ck_dir}")
                maybe = load_vq_wrapper(
                    base_model_name=cfg.model.pretrained_name,
                    layer_after=int(vq_cfg.bottleneck.layer_after),
                    codebook_size=int(K),
                    beta=float(vq_cfg.bottleneck.commitment_beta),
                    ckpt_dir=ck_dir,
                )
                if maybe is None:
                    log.warning("[VQ] Cache load failed, retraining.")
                model = maybe if maybe is not None else None
            else:
                model = None

            if model is None:
                log.info(f"[VQ] Training VQ model for K={K}")
                model, vq_tok = train_vq_joint(
                    base_model_name=cfg.model.pretrained_name,
                    dataset_name=cfg.data.source.hf_dataset,
                    dataset_config=cfg.data.source.hf_config,
                    text_field=text_field,
                    max_length=cfg.data.processing.max_length,
                    layer_after=int(vq_cfg.bottleneck.layer_after),
                    codebook_size=int(K),
                    lr=5e-5,
                    epochs=int(getattr(vq_cfg.index_lm, "epochs", 2)),
                    beta=float(vq_cfg.bottleneck.commitment_beta),
                )
                out_dir = cache_dir_for(cfg.model.name, int(K))
                meta = {
                    "base_model_name": cfg.model.pretrained_name,
                    "layer_after": int(vq_cfg.bottleneck.layer_after),
                    "codebook_size": int(K),
                    "beta": float(vq_cfg.bottleneck.commitment_beta),
                }
                save_vq_wrapper(model, out_dir, meta)
                log.info(f"[VQ] Saved cached wrapper -> {out_dir / 'model.pt'}")

            device = next(model.parameters()).device

            all_indices: List[List[int]] = []
            all_texts: List[str] = []
            for ex in split:
                text = ex[text_field]
                ids = tok(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=cfg.data.processing.max_length,
                )["input_ids"].to(device)
                idx = encode_indices(model, ids)[0].tolist()
                all_indices.append(idx)
                all_texts.append(text)

            lm = train_index_lm(
                all_indices,
                K=int(K),
                hidden=int(vq_cfg.index_lm.hidden_size),
                layers=int(vq_cfg.index_lm.layers),
                epochs=int(vq_cfg.index_lm.epochs),
            )

            lm_cpu = type(lm)(
                vocab_size=int(K),
                hidden_size=int(vq_cfg.index_lm.hidden_size),
                layers=int(vq_cfg.index_lm.layers),
            )
            lm_cpu.load_state_dict(lm.state_dict(), strict=True)
            lm_cpu.to(torch.device("cpu")).eval()

            total_bits_payload = 0.0
            total_tokens = 0
            total_chars = 0

            payload_dir = payload_root / f"vq_{cfg.model.pretrained_name}_K{K}"
            payload_dir.mkdir(parents=True, exist_ok=True)
            dump_path = payload_dir / "recons.jsonl"
            if dump_path.exists():
                dump_path.unlink()

            charF_scores: List[float] = []
            chrf_scores: List[float] = []
            berts_scores: List[float] = []
            sem_scores: List[float] = []
            tcm_vals: List[float] = []
            pcm_vals: List[float] = []
            cpu_decode_ms: List[float] = []

            for doc_id, (idx, text) in enumerate(zip(all_indices, all_texts)):
                if constriction is not None and len(idx) >= 2:
                    try:
                        payload = _encode_indices_arithmetic_cpu(lm_cpu, idx)
                        token_bits = len(payload) * 8
                        t0 = time.perf_counter()
                        _ = _decode_indices_arithmetic_cpu(
                            lm_cpu, payload, first_sym=int(idx[0]), length=len(idx) - 1
                        )
                        cpu_decode_ms.append((time.perf_counter() - t0) * 1000.0)
                    except Exception:
                        token_bits = cross_entropy_bits_index_stream(lm, idx)
                else:
                    token_bits = cross_entropy_bits_index_stream(lm, idx)

                total_bits_payload += float(token_bits)
                total_tokens += max(0, len(idx) - 1)
                total_chars += len(text)

                out_ids = reconstruct_tokens_from_indices(
                    model,
                    idx,
                    start_token_id=tok.bos_token_id or tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                    margin_factor=float(
                        cfg.experiment.stage3.stopping.eos_or_maxlen_margin
                    ),
                )
                recon_text = tok.decode(out_ids, skip_special_tokens=True)
                charF = character_level_fidelity(text, recon_text)
                chrF = chrf_score(text, recon_text, order=6)
                bF1 = bertscore_f1(text, recon_text, model_type="roberta-large")
                sspan = semantic_span_fidelity(text, recon_text)

                charF_scores.append(charF)
                chrf_scores.append(chrF)
                berts_scores.append(bF1)
                sem_scores.append(sspan)

                if doc_id % 10 == 0:
                    s0 = cp_oracle.surprisal_bits(text)[1]
                    s1 = cp_oracle.surprisal_bits(recon_text)[1]
                    m = tcm_pcm_from_surprisal(s0, s1, vocab_size=len(tok))
                    tcm_vals.append(float(m["tcm_bits"]))
                    pcm_vals.append(float(m["pcm_bits"]))

                record = {
                    "doc_id": doc_id,
                    "codebook_size": int(K),
                    "position_bits": 0,
                    "token_bits": int(token_bits),
                    "orig_chars": len(text),
                    "cpu_decode_ms": (cpu_decode_ms[-1] if cpu_decode_ms else None),
                    "original": text,
                    "reconstruction": recon_text,
                    "char_fidelity": charF,
                    "chrf": chrF,
                    "bertscore_f1": bF1,
                    "semantic_span_fid": sspan,
                }
                with open(dump_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            bpt = total_bits_payload / max(1, total_tokens)
            bpc = total_bits_payload / max(1, total_chars)
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
            cpu_decode_mean = (
                float(sum(cpu_decode_ms) / max(1, len(cpu_decode_ms)))
                if cpu_decode_ms
                else None
            )
            cpu_decode_std = float(np.std(cpu_decode_ms)) if cpu_decode_ms else None

            point = {
                "method": "VQ",
                "codebook_size": int(K),
                "bpc": bpc,
                "bpt": bpt,
                "charF_mean": char_fid_point,
                "chrf_mean": chrf_point,
                "bertscore_f1_mean": berts_point,
                "sem_span_fid_mean": sem_point,
                "tcm_mean": tcm_point,
                "pcm_mean": pcm_point,
                "cpu_decode_ms_mean": cpu_decode_mean,
                "cpu_decode_ms_std": cpu_decode_std,
            }
            points.append(point)
            rd_file.write_text(json.dumps(points, indent=2))

            wandb_log.log(
                {
                    "vq/K": int(K),
                    "vq/bpc": float(bpc),
                    "vq/bpt": float(bpt),
                    "vq/char_fid": float(char_fid_point),
                    "vq/chrf": float(chrf_point),
                    "vq/bertscore_f1": float(berts_point),
                    "vq/sem_span_fid": float(sem_point),
                    "vq/tcm": float(tcm_point),
                    "vq/pcm": float(pcm_point),
                    **(
                        {
                            "vq/cpu_decode_ms_mean": cpu_decode_mean,
                            "vq/cpu_decode_ms_std": cpu_decode_std,
                        }
                        if cpu_decode_mean is not None
                        else {}
                    ),
                }
            )

        wandb_log.finish()

    _run()


if __name__ == "__main__":
    main()
