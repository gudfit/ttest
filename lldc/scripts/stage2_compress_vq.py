from __future__ import annotations
from typing import Any, List, Dict
import json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from lldc.utils.logging import setup_logging
from lldc.utils.hydra_utils import resolve_auto
from lldc.utils.paths import Paths
from lldc.utils import wandb_log
from lldc.metrics.fidelity import character_level_fidelity
from lldc.metrics.crumpled_paper import Oracle as AROracle, tcm_pcm_from_surprisal
from lldc.models.vq.vq_trainer import (
    train_vq_joint,
    encode_indices,
    train_index_lm,
    cross_entropy_bits_index_stream,
    VQBottleneckWrapper,
)
from lldc.decompression.vq_decompress import reconstruct_tokens_from_indices
from lldc.models.vq.cache import cache_dir_for, cached_checkpoint


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

        reuse_cache = bool(
            getattr(getattr(cfg, "stage2", {}), "vq", {}).get("reuse_cache", True)
        )
        force_retrain = bool(
            getattr(getattr(cfg, "stage2", {}), "vq", {}).get("force_retrain", False)
        )

        for K in Ks:
            ck = (
                cached_checkpoint(cfg.model.name, int(K))
                if (reuse_cache and not force_retrain)
                else None
            )
            if ck is not None:
                log.info(f"[VQ] Reusing cached VQ model for K={K}: {ck}")
                model = VQBottleneckWrapper.from_pretrained(
                    base_model_name, K=int(K), ckpt_path=str(ck)
                )
            else:
                log.info(f"[VQ] Training VQ model for K={K}")
                model, vq_tok = train_vq_joint(
                    base_model_name=cfg.model.pretrained_name,
                    dataset_name=cfg.data.source.hf_dataset,
                    dataset_config=cfg.data.source.hf_config,
                    text_field=text_field,
                    max_length=cfg.data.processing.max_length,
                    layer_after=int(cfg.experiment.stage2.vq.bottleneck.layer_after),
                    codebook_size=int(K),
                    lr=5e-5,
                    epochs=int(getattr(cfg.experiment.stage2.vq.index_lm, "epochs", 2)),
                    beta=float(cfg.experiment.stage2.vq.bottleneck.commitment_beta),
                )
                out_dir = cache_dir_for(cfg.model.name, int(K))
                out_dir.mkdir(parents=True, exist_ok=True)
                save_path = out_dir / "best.safetensors"
                model.save_pretrained(str(save_path))
                log.info(f"[VQ] Saved cached model -> {save_path}")

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
                hidden=int(cfg.experiment.stage2.vq.index_lm.hidden_size),
                layers=int(cfg.experiment.stage2.vq.index_lm.layers),
                epochs=int(cfg.experiment.stage2.vq.index_lm.epochs),
            )
            total_bits = 0.0
            total_tokens = 0
            total_chars = 0

            payload_dir = payload_root / f"vq_{cfg.model.pretrained_name}_K{K}"
            payload_dir.mkdir(parents=True, exist_ok=True)
            dump_path = payload_dir / "recons.jsonl"
            if dump_path.exists():
                dump_path.unlink()

            charF_scores: List[float] = []
            tcm_vals: List[float] = []
            pcm_vals: List[float] = []

            for doc_id, (idx, text) in enumerate(zip(all_indices, all_texts)):
                total_bits += cross_entropy_bits_index_stream(lm, idx)
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
                charF_scores.append(character_level_fidelity(text, recon_text))

                if doc_id % 10 == 0:
                    s0 = cp_oracle.surprisal_bits(text)
                    s1 = cp_oracle.surprisal_bits(recon_text)
                    m = tcm_pcm_from_surprisal(s0, s1, vocab_size=len(tok))
                    tcm_vals.append(float(m["tcm_bits"]))
                    pcm_vals.append(float(m["pcm_bits"]))

                with open(dump_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "doc_id": doc_id,
                                "codebook_size": int(K),
                                "original": text,
                                "reconstruction": recon_text,
                            }
                        )
                        + "\n"
                    )

            bpt = total_bits / max(1, total_tokens)
            bpc = total_bits / max(1, total_chars)
            char_fid_point = float(sum(charF_scores) / max(1, len(charF_scores)))
            tcm_point = (
                float(sum(tcm_vals) / max(1, len(tcm_vals))) if tcm_vals else 0.0
            )
            pcm_point = (
                float(sum(pcm_vals) / max(1, len(pcm_vals))) if pcm_vals else 0.0
            )

            point = {
                "method": "VQ",
                "codebook_size": int(K),
                "bpc": bpc,
                "bpt": bpt,
                "charF_mean": char_fid_point,
                "tcm_mean": tcm_point,
                "pcm_mean": pcm_point,
            }
            points.append(point)
            rd_file.write_text(json.dumps(points, indent=2))

            wandb_log.log(
                {
                    "vq/K": int(K),
                    "vq/bpc": float(bpc),
                    "vq/char_fid": float(char_fid_point),
                    "vq/tcm": float(tcm_point),
                    "vq/pcm": float(pcm_point),
                }
            )

        wandb_log.finish()

    _run()


if __name__ == "__main__":
    main()
