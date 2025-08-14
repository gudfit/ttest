# lldc/scripts/stage2_compress_vq.py

from __future__ import annotations
from typing import Any, List, Dict, Tuple
import json, math, time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

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

from lldc.compression.payload_codec import arithmetic as ac
from lldc.models.vq.vq_trainer import (
    VQBottleneckWrapper,
    IndexLM,
    train_index_lm,
    encode_indices,
)
from lldc.models.vq.cache import cached_checkpoint, cache_dir_for
from lldc.models.vq.vq_trainer import load_vq_wrapper


def _load_vq_model(cfg: Any) -> Tuple[VQBottleneckWrapper, AutoTokenizer]:
    base_name = cfg.model.pretrained_name
    K = int(cfg.experiment.stage2.vq.codebook_sizes[0])
    layer_after = int(cfg.experiment.stage2.vq.bottleneck.layer_after)
    beta = float(cfg.experiment.stage2.vq.bottleneck.commitment_beta)

    ck = cached_checkpoint(base_name, K)
    if ck is None:
        raise RuntimeError(
            f"No cached VQ checkpoint found for base='{base_name}', K={K}. "
            f"Train it first, then rerun stage-2. Expected under {cache_dir_for(base_name, K)}"
        )
    model = load_vq_wrapper(
        base_model_name=base_name,
        layer_after=layer_after,
        codebook_size=K,
        beta=beta,
        ckpt_dir=ck,
    )
    if model is None:
        raise RuntimeError("Failed to load VQ wrapper checkpoint.")
    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def _indexlm_probs_for_sequence(
    lm: IndexLM, seq: List[int], K: int
) -> List[List[float]]:
    device = next(lm.parameters()).device
    if not seq:
        return []
    start_id = 0
    x = torch.tensor([start_id] + seq[:-1], dtype=torch.long, device=device).unsqueeze(
        0
    )
    logits = lm(x)[0]
    probs = torch.softmax(logits, dim=-1)
    return [p.detach().cpu().tolist() for p in probs]


def main():
    import hydra

    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        cfg = resolve_auto(cfg)
        paths = Paths().ensure()

        set_determinism(getattr(cfg, "seed", 13))
        run = wandb_log.start(
            cfg,
            run_name=f"S2-VQ-{cfg.model.name}-seed{getattr(cfg,'seed',13)}",
            tags=["stage2", "vq"],
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        vq_model, tok = _load_vq_model(cfg)
        vq_model.to(device).eval()

        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        split = ds[cfg.data.source.split_map.test]
        text_field = cfg.data.processing.text_field
        K = vq_model.vq.codebook_size
        all_indices_for_lm: List[List[int]] = []
        max_for_lm = int(
            getattr(cfg.experiment.stage2.vq.index_lm, "train_samples", 1000)
        )
        for i, ex in enumerate(split.select(range(min(max_for_lm, len(split))))):
            txt = ex[text_field]
            ids = tok(
                txt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.data.processing.max_length,
                add_special_tokens=False,
            )["input_ids"].to(device)
            idx = encode_indices(vq_model, ids)[0].tolist()
            if idx:
                all_indices_for_lm.append(idx)
        lm = train_index_lm(
            all_indices_for_lm,
            K=K,
            hidden=int(cfg.experiment.stage2.vq.index_lm.hidden_size),
            layers=int(cfg.experiment.stage2.vq.index_lm.layers),
            epochs=int(cfg.experiment.stage2.vq.index_lm.epochs),
        )
        lm.eval()

        seed = getattr(cfg, "seed", 13)
        payload_dir = paths.payloads / f"vq_{cfg.model.pretrained_name}_K{K}_seed{seed}"
        payload_dir.mkdir(parents=True, exist_ok=True)
        dump_path = payload_dir / "recons.jsonl"
        if dump_path.exists():
            dump_path.unlink()

        cp_oracle = AROracle("gpt2-large", device=device)

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
            enc = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.data.processing.max_length,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            idx = encode_indices(vq_model, input_ids)[0].tolist()
            kept_count = len(idx)
            token_bits = 0
            token_decode_ms = 0.0
            token_payload_status = "none"
            if kept_count > 0:
                try:
                    probs = _indexlm_probs_for_sequence(lm, idx, K)
                    payload = ac.encode_with_probs(idx, probs)
                    token_bits = ac.payload_num_bits(payload)
                    t0 = time.perf_counter()
                    _ = ac.decode_with_probs(payload, probs)
                    token_decode_ms = (time.perf_counter() - t0) * 1000.0
                    token_payload_status = "encoded_arithmetic"
                except Exception:
                    with torch.no_grad():
                        probs = _indexlm_probs_for_sequence(lm, idx, K)
                        eps = 1e-12
                        xb = 0.0
                        for sym, p in zip(idx, probs):
                            pr = float(p[sym]) if 0 <= sym < len(p) else 0.0
                            xb += -math.log2(max(pr, eps))
                        token_bits = int(round(xb))
                        token_payload_status = "estimated_xent"

            cpu_decode_ms.append(token_decode_ms)
            with torch.no_grad():
                idx_t = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)
                pred_ids = vq_model.decode_from_indices(idx_t)[0]
                recon_text = tok.decode(pred_ids.tolist(), skip_special_tokens=True)

            charF = character_level_fidelity(text, recon_text)
            chrF = chrf_score(
                text, recon_text, order=int(getattr(cfg.fidelity.chrf, "order", 6))
            )
            bF1 = bertscore_f1(text, recon_text, model_type="roberta-large")
            sspan = semantic_span_fidelity(text, recon_text)

            charF_scores.append(charF)
            chrf_scores.append(chrF)
            berts_scores.append(bF1)
            sem_scores.append(sspan)

            if doc_id % 10 == 0:
                s_orig = cp_oracle.surprisal_bits(text)[1]
                s_rec = cp_oracle.surprisal_bits(recon_text)[1]
                m = tcm_pcm_from_surprisal(s_orig, s_rec, vocab_size=len(tok))
                tcm_vals.append(float(m["tcm_bits"]))
                pcm_vals.append(float(m["pcm_bits"]))

            with open(dump_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "position_bits": 0,
                            "token_bits": token_bits,
                            "kept_tokens": kept_count,
                            "token_payload_status": token_payload_status,
                            "orig_chars": len(text),
                            "original": text,
                            "reconstruction": recon_text,
                            "char_fidelity": charF,
                            "chrf": chrF,
                            "bertscore_f1": bF1,
                            "semantic_span_fid": sspan,
                            "cpu_decode_ms": token_decode_ms,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            totals["token_bits"] += token_bits
            totals["kept_tokens"] += kept_count
            totals["chars"] += len(text)

        bpc = (totals["position_bits"] + totals["token_bits"]) / max(1, totals["chars"])
        bpt = (
            totals["token_bits"] / max(1, totals["kept_tokens"])
            if totals["kept_tokens"] > 0
            else None
        )
        char_fid_point = float(sum(charF_scores) / max(1, len(charF_scores)))
        chrf_point = float(sum(chrf_scores) / max(1, len(chrf_scores)))
        berts_point = float(sum(berts_scores) / max(1, len(berts_scores)))
        sem_point = float(sum(sem_scores) / max(1, len(sem_scores)))
        tcm_point = float(sum(tcm_vals) / max(1, len(tcm_vals))) if tcm_vals else 0.0
        pcm_point = float(sum(pcm_vals) / max(1, len(pcm_vals))) if pcm_vals else 0.0
        cpu_decode_mean = (
            float(sum(cpu_decode_ms) / max(1, len(cpu_decode_ms)))
            if cpu_decode_ms
            else 0.0
        )

        point = {
            "method": "VQ",
            "codebook_K": int(K),
            "bpc": float(bpc),
            "bpt": float(bpt) if bpt is not None else None,
            "charF_mean": char_fid_point,
            "chrf_mean": chrf_point,
            "bertscore_f1_mean": berts_point,
            "sem_span_fid_mean": sem_point,
            "tcm_mean": tcm_point,
            "pcm_mean": pcm_point,
            "cpu_decode_ms_mean": cpu_decode_mean,
        }
        (paths.rd_curves / "vq_points.json").write_text(json.dumps([point], indent=2))
        (paths.results / "vq_points.json").write_text(json.dumps([point], indent=2))

        wandb_log.log(
            {
                "vq/payload_bits": int(totals["token_bits"]),
                "vq/bpc": float(bpc),
                "vq/bpt": float(bpt) if bpt is not None else 0.0,
                "vq/char_fid": float(char_fid_point),
                "vq/chrf": float(chrf_point),
                "vq/bertscore_f1": float(berts_point),
                "vq/sem_span_fid": float(sem_point),
                "vq/tcm": float(tcm_point),
                "vq/pcm": float(pcm_point),
                "vq/cpu_decode_ms_mean": float(cpu_decode_mean),
            }
        )
        wandb_log.finish()

    _run()


if __name__ == "__main__":
    main()
