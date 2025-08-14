# lldc/scripts/stage1_specialise.py
from __future__ import annotations
import os, json, logging, math, time
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_dataset, Dataset, DatasetDict

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.utils.seed import resolve_seeds
from lldc.utils.determinism import set_determinism
from lldc.utils.hydra_utils import resolve_auto
from lldc.utils import wandb_log
from lldc.compression.predictive_masking import pll_surprisal_scores


@dataclass
class _Stage1Cfg:
    policy_set_fraction: float = 0.10
    recompute_each_epoch: bool = True
    rate_start: float = 0.2
    rate_end: float = 0.8
    pll_token_scoring: bool = True


def _read_s1_mask_cfg(cfg: Any) -> _Stage1Cfg:
    s1 = getattr(getattr(cfg, "experiment", None), "stage1", None) or getattr(
        cfg, "stage1", None
    )
    m = getattr(s1, "masking", None) if s1 else None
    return _Stage1Cfg(
        policy_set_fraction=(
            float(getattr(m, "policy_set_fraction", 0.10)) if m else 0.10
        ),
        recompute_each_epoch=(
            bool(getattr(m, "recompute_each_epoch", True)) if m else True
        ),
        rate_start=(
            float(getattr(getattr(m, "curriculum", None), "rate_start", 0.2))
            if m
            else 0.2
        ),
        rate_end=(
            float(getattr(getattr(m, "curriculum", None), "rate_end", 0.8))
            if m
            else 0.8
        ),
        pll_token_scoring=bool(getattr(m, "pll_token_scoring", True)) if m else True,
    )


def _resolve_precision(cfg: Any) -> str:
    hw = getattr(getattr(cfg, "compute", None), "hardware", None)
    hw_dtype = (getattr(hw, "dtype", None) or "").lower() if hw else ""
    mp = (getattr(getattr(cfg, "compute", None), "mixed_precision", None) or "").lower()
    dtype = hw_dtype or mp
    if dtype not in {"bf16", "fp16", "fp32"}:
        dtype = "fp32"
    if not torch.cuda.is_available():
        return "fp32"
    return dtype


class AdaptiveMaskingCollator:

    def __init__(self, tokenizer, model, threshold_bits: float, mask_rate: float):
        self.tok = tokenizer
        self.model = model
        self.threshold_bits = float(threshold_bits)
        self.mask_rate = float(mask_rate)

    def update_policy(self, threshold_bits: float, mask_rate: float):
        self.threshold_bits = float(threshold_bits)
        self.mask_rate = float(mask_rate)

    def __call__(self, examples: List[dict]) -> dict:
        input_ids_list = [
            torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples
        ]
        device = next(self.model.parameters()).device
        mask_id = int(self.tok.mask_token_id)
        input_ids_list = [ids[: self.tok.model_max_length] for ids in input_ids_list]

        masked_inputs: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        with torch.no_grad():
            for ids in input_ids_list:
                ids = ids.to(device)
                s_bits = pll_surprisal_scores(ids, self.model, self.tok, mask_id)
                to_mask = s_bits <= self.threshold_bits
                L = ids.size(0)
                target_m = max(1, int(round(self.mask_rate * L)))
                cur_m = int(to_mask.sum().item())
                if cur_m != target_m and L > 0:
                    k = min(max(1, target_m), L)
                    order = torch.argsort(s_bits, dim=0)
                    to_mask = torch.zeros_like(to_mask)
                    to_mask[order[:k]] = True

                masked = ids.clone()
                masked[to_mask] = mask_id
                labels = torch.full_like(ids, fill_value=-100)
                labels[to_mask] = ids[to_mask]

                masked_inputs.append(masked)
                labels_list.append(labels)

        batch_input = torch.nn.utils.rnn.pad_sequence(
            masked_inputs, batch_first=True, padding_value=self.tok.pad_token_id or 0
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attn_mask = (batch_input != (self.tok.pad_token_id or 0)).long()
        return {
            "input_ids": batch_input,
            "labels": batch_labels,
            "attention_mask": attn_mask,
        }


def _linear_curriculum(
    epoch_idx: int, num_epochs: int, rate_start: float, rate_end: float
) -> float:
    if num_epochs <= 1:
        return float(rate_end)
    alpha = epoch_idx / float(max(1, num_epochs - 1))
    return float(rate_start + (rate_end - rate_start) * alpha)


@torch.no_grad()
def _compute_global_threshold_bits(
    model,
    tok,
    ds_policy: Dataset,
    target_mask_rate: float,
    max_examples: int = 1000,
) -> float:
    device = next(model.parameters()).device
    mask_id = int(tok.mask_token_id)
    all_scores: List[float] = []
    n = min(max_examples, len(ds_policy))
    for i in range(n):
        ids = torch.tensor(ds_policy[i]["input_ids"], dtype=torch.long, device=device)
        ids = ids[: tok.model_max_length]
        s = pll_surprisal_scores(ids, model, tok, mask_id)
        all_scores.extend([float(x.item()) for x in s])
    if not all_scores:
        return 0.0
    q = np.quantile(np.array(all_scores, dtype=np.float32), target_mask_rate)
    return float(q)


class PolicyRescoreCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        ds_policy_tok: Dataset,
        s1cfg: _Stage1Cfg,
        collator: AdaptiveMaskingCollator,
        max_examples: int = 600,
    ):
        self.tok = tokenizer
        self.ds_policy_tok = ds_policy_tok
        self.s1cfg = s1cfg
        self.collator = collator
        self.max_examples = max_examples

    def on_epoch_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return
        if not self.s1cfg.recompute_each_epoch:
            return
        epoch_idx = int(state.epoch) if state.epoch is not None else 0
        num_epochs = int(args.num_train_epochs)
        mask_rate = _linear_curriculum(
            epoch_idx, num_epochs, self.s1cfg.rate_start, self.s1cfg.rate_end
        )
        thresh = _compute_global_threshold_bits(
            trainer.model,
            self.tok,
            self.ds_policy_tok,
            target_mask_rate=mask_rate,
            max_examples=self.max_examples,
        )
        self.collator.update_policy(threshold_bits=thresh, mask_rate=mask_rate)


def _tokenize_dataset(
    tok, ds: DatasetDict, text_field: str, max_length: int
) -> DatasetDict:
    def _tok_fn(batch):
        return tok(
            batch[text_field],
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

    return ds.map(_tok_fn, batched=True, remove_columns=[text_field])


def main() -> None:
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        cfg = resolve_auto(cfg)
        paths = Paths().ensure()

        seeds = resolve_seeds(
            getattr(cfg, "seed", None), getattr(cfg, "num_runs", None)
        )
        seed = int(seeds[0])
        set_determinism(
            seed,
            cudnn_benchmark=getattr(
                getattr(cfg, "repro", {}), "cudnn_benchmark", False
            ),
            allow_tf32=getattr(getattr(cfg, "repro", {}), "allow_tf32", True),
        )

        model_name = cfg.model.pretrained_name
        if cfg.model.arch != "mlm":
            log.info(f"[stage1] Skipping specialisation for non-MLM model={model_name}")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.mask_token is None:
            raise RuntimeError(f"Tokenizer for {model_name} must have a [MASK] token.")
        model = AutoModelForMaskedLM.from_pretrained(model_name)

        prec = _resolve_precision(cfg)
        if torch.cuda.is_available():
            if prec == "bf16":
                model.to(torch.bfloat16)
            elif prec == "fp16":
                model.to(torch.float16)
        model.to(device)

        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        text_field = cfg.data.processing.text_field

        full_train = ds[cfg.data.source.split_map.train]
        split = full_train.train_test_split(test_size=0.10, seed=seed)
        fine_set_raw: Dataset = split["train"]
        policy_set_raw: Dataset = split["test"]

        ds_train_tok = _tokenize_dataset(
            tok,
            DatasetDict({"train": fine_set_raw}),
            text_field,
            cfg.data.processing.max_length,
        )["train"]
        ds_policy_tok = _tokenize_dataset(
            tok,
            DatasetDict({"policy": policy_set_raw}),
            text_field,
            cfg.data.processing.max_length,
        )["policy"]
        ds_valid_tok = _tokenize_dataset(
            tok,
            DatasetDict({"valid": ds[cfg.data.source.split_map.valid]}),
            text_field,
            cfg.data.processing.max_length,
        )["valid"]

        s1cfg = _read_s1_mask_cfg(cfg)
        init_thresh = _compute_global_threshold_bits(
            model,
            tok,
            ds_policy_tok,
            target_mask_rate=s1cfg.rate_start,
            max_examples=600,
        )
        collator = AdaptiveMaskingCollator(
            tok, model, threshold_bits=init_thresh, mask_rate=s1cfg.rate_start
        )

        args = TrainingArguments(
            output_dir=str(paths.checkpoints / f"{model_name}_stage1_seed{seed}"),
            per_device_train_batch_size=cfg.data.loader.batch_size or 8,
            per_device_eval_batch_size=cfg.data.loader.batch_size or 8,
            gradient_accumulation_steps=cfg.compute.gradient_accumulation or 1,
            num_train_epochs=(
                cfg.model.specialise.epochs
                if cfg.model.specialise.epochs != "auto"
                else 4
            ),
            learning_rate=(
                cfg.model.specialise.lr if cfg.model.specialise.lr != "auto" else 5e-5
            ),
            weight_decay=cfg.model.specialise.weight_decay,
            warmup_ratio=cfg.model.specialise.warmup_ratio,
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            fp16=(prec == "fp16" and torch.cuda.is_available()),
            bf16=(prec == "bf16" and torch.cuda.is_available()),
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_train_tok,
            eval_dataset=ds_valid_tok,
            data_collator=collator,
            tokenizer=tok,
        )

        trainer.add_callback(PolicyRescoreCallback(tok, ds_policy_tok, s1cfg, collator))

        run = wandb_log.start(
            cfg,
            run_name=f"S1-{cfg.model.name}-seed{seed}",
            tags=["stage1", "specialise"],
        )
        trainer.train()
        wandb_log.finish()

        ckpt_dir = Paths().checkpoints / f"{model_name}_stage1_seed{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tok.save_pretrained(ckpt_dir)
        (ckpt_dir / "done.txt").write_text("ok\n")
        logging.getLogger("lldc").info(f"[stage1] Done. Checkpoint at: {ckpt_dir}")

    _run()


if __name__ == "__main__":
    main()
