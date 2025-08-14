# lldc/scripts/stage2_compress_vq.py

from __future__ import annotations
from typing import Any, List, Dict, Tuple
import json, math, time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM as _AutoMLM,
)

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

from lldc.metrics.crumpled_paper import (
    OracleEnsemble,
    tcm_pcm_from_surprisal,
    delta_log_likelihood_bits,
)

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
    if not seq:
        return []
    device = next(lm.parameters()).device
    start_id = 0
    x = torch.tensor([start_id] + seq[:-1], dtype=torch.long, device=device).unsqueeze(
        0
    )
    logits = lm(x)[0]
    probs = F.softmax(logits, dim=-1)
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

        ensemble_list = list(
            getattr(cfg, "crumpled_paper", {}).get("oracle_models", ["gpt2-large"])
        )
        measuring_tok_cfg = getattr(cfg, "crumpled_paper", {}).get(
            "measuring_tokenizer", "oracle"
        )
        cp_ensemble = OracleEnsemble(
            model_names=ensemble_list,
            device=device,
            measuring_tokenizer=measuring_tok_cfg,
        )

        bi_name = getattr(cfg, "crumpled_paper", {}).get(
            "delta_ll_model", "roberta-large"
        )
        bi_tok = AutoTokenizer.from_pretrained(bi_name)
        bi_m = _AutoMLM.from_pretrained(bi_name).to(device).eval()

        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        split = ds[cfg.data.source.split_map.test]
        text_field = cfg.data.processing.text_field
        K = int(vq_model.vq.codebook_size)
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

        lm = (
            train_index_lm(
                all_indices_for_lm,
                K=K,
                hidden=int(cfg.experiment.stage2.vq.index_lm.hidden_size),
                layers=int(cfg.experiment.stage2.vq.index_lm.layers),
                epochs=int(cfg.experiment.stage2.vq.index_lm.epochs),
            )
            .to(device)
            .eval()
        )

        seed = getattr(cfg, "seed", 13)
        payload_dir = paths.payloads / f"vq_{cfg.model.pretrained_name}_seed{seed}"
        payload_dir.mkdir(parents=True, exist_ok=True)
        dump_path = payload_dir / "recons.jsonl"
        if dump_path.exists():
            dump_path.unlink()

        totals = {
            "token_bits": 0,
            "chars": 0,
            "n_docs": 0,
        }
        charF_scores: List[float] = []
        chrf_scores: List[float] = []
        berts_scores: List[float] = []
        sem_scores: List[float] = []
        tcm_vals: List[float] = []
        pcm_vals: List[float] = []
        dll_vals: List[float] = []
        cpu_decode_ms: List[float] = []

        for doc_id, ex in enumerate(split):
            text = ex[text_field]
            ids = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.data.processing.max_length,
                add_special_tokens=False,
            )["input_ids"].to(device)
            idx = encode_indices(vq_model, ids)[0].tolist()
            probs = _indexlm_probs_for_sequence(lm, idx, K)
            token_bits = 0
            t_decode = 0.0
            if idx:
                try:
                    payload = ac.encode_with_probs(idx, probs)
                    token_bits = ac.payload_num_bits(payload)
                    t0 = time.perf_counter()
                    _ = ac.decode_with_probs(payload, probs)
                    t_decode = (time.perf_counter() - t0) * 1000.0
                except Exception:
                    xb = 0.0
                    eps = 1e-12
                    for sym, p in zip(idx, probs):
                        pr = float(p[sym]) if 0 <= sym < len(p) else 0.0
                        xb += -math.log2(max(pr, eps))
                    token_bits = int(round(xb))
                    t_decode = 0.0

            totals["token_bits"] += token_bits
            totals["chars"] += len(text)
            totals["n_docs"] += 1
            cpu_decode_ms.append(t_decode)
            with torch.no_grad():
                recon_ids = vq_model.decode_from_indices(
                    torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)
                )[0]
            recon_text = tok.decode(recon_ids, skip_special_tokens=True)
            charF = character_level_fidelity(text, recon_text)
            chrF = chrf_score(text, recon_text, order=6)
            bF1 = bertscore_f1(text, recon_text, model_type="roberta-large")
            sspan = semantic_span_fidelity(text, recon_text)

            charF_scores.append(charF)
            chrf_scores.append(chrF)
            berts_scores.append(bF1)
            sem_scores.append(sspan)
            if doc_id % 10 == 0:
                s_orig = cp_ensemble.surprisal_bits(text)[1]
                s_rec = cp_ensemble.surprisal_bits(recon_text)[1]
                m = tcm_pcm_from_surprisal(
                    s_orig, s_rec, vocab_size=cp_ensemble.vocab_size
                )
                tcm_vals.append(float(m["tcm_bits"]))
                pcm_vals.append(float(m["pcm_bits"]))
                try:
                    dll = delta_log_likelihood_bits(text, recon_text, bi_m, bi_tok)
                    dll_vals.append(float(dll))
                except Exception:
                    pass

            with open(dump_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "original": text,
                            "reconstruction": recon_text,
                            "token_bits": token_bits,
                            "char_fidelity": charF,
                            "chrf": chrF,
                            "bertscore_f1": bF1,
                            "semantic_span_fid": sspan,
                            "cpu_decode_ms": t_decode,
                            "delta_ll_bits": (
                                (dll_vals[-1] if dll_vals else None)
                                if (doc_id % 10 == 0)
                                else None
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        bpc = totals["token_bits"] / max(1, totals["chars"])
        charF_mean = float(sum(charF_scores) / max(1, len(charF_scores)))
        chrf_mean = float(sum(chrf_scores) / max(1, len(chrf_scores)))
        berts_mean = float(sum(berts_scores) / max(1, len(berts_scores)))
        sem_mean = float(sum(sem_scores) / max(1, len(sem_scores)))
        tcm_mean = float(sum(tcm_vals) / max(1, len(tcm_vals))) if tcm_vals else 0.0
        pcm_mean = float(sum(pcm_vals) / max(1, len(pcm_vals))) if pcm_vals else 0.0
        dll_mean = float(sum(dll_vals) / max(1, len(dll_vals))) if dll_vals else 0.0
        cpu_decode_ms_mean = float(sum(cpu_decode_ms) / max(1, len(cpu_decode_ms)))

        point = {
            "method": "VQ",
            "codebook_K": K,
            "bpc": float(bpc),
            "bpt": None,
            "charF_mean": charF_mean,
            "chrf_mean": chrf_mean,
            "bertscore_f1_mean": berts_mean,
            "sem_span_fid_mean": sem_mean,
            "tcm_mean": tcm_mean,
            "pcm_mean": pcm_mean,
            "delta_ll_bits_mean": dll_mean,
            "cpu_decode_ms_mean": cpu_decode_ms_mean,
            "n_docs": int(totals["n_docs"]),
        }
        (paths.rd_curves / "vq_points.json").write_text(json.dumps([point], indent=2))
        wandb_log.log(
            {"vq/bpc": float(bpc), "vq/charF": charF_mean, "vq/delta_ll_bits": dll_mean}
        )
        wandb_log.finish()

    _run()


if __name__ == "__main__":
    main()
