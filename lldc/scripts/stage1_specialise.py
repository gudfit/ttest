from __future__ import annotations
import os, json, logging
from dataclasses import dataclass
from typing import Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

from lldc.utils.logging import setup_logging
from lldc.utils.paths import Paths
from lldc.utils.seed import resolve_seeds
from lldc.utils.determinism import set_determinism
from lldc.utils.hydra_utils import resolve_auto
from lldc.utils import wandb_log


@dataclass
class Cfg:
    pass


def _auto_device_dtype(cfg) -> tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = cfg.compute.hardware.dtype
    if dt == "bf16":
        torch_dtype = torch.bfloat16
    elif dt == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    return device, torch_dtype


def main() -> None:
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        cfg = resolve_auto(cfg)
        paths = Paths().ensure()
        seeds = resolve_seeds(cfg.seed, cfg.num_runs)
        device, torch_dtype = _auto_device_dtype(cfg)

        model_name = cfg.model.pretrained_name
        if cfg.model.arch != "mlm":
            log.info(f"[stage1] Skipping specialisation for non-MLM model={model_name}")
            return

        log.info(
            f"[stage1] Specialising {model_name} on {cfg.data.name} ({device}, {torch_dtype})"
        )
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
        model.to(device)

        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        text_field = cfg.data.processing.text_field

        def tok_fn(batch):
            return tok(
                batch[text_field],
                truncation=True,
                max_length=cfg.data.processing.max_length,
            )

        ds = ds.map(tok_fn, batched=True, remove_columns=[text_field])
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tok, mlm=True, mlm_probability=0.15
        )

        args = TrainingArguments(
            output_dir=str(paths.checkpoints / f"{model_name}_stage1"),
            per_device_train_batch_size=cfg.data.loader.batch_size or 8,
            per_device_eval_batch_size=cfg.data.loader.batch_size or 8,
            gradient_accumulation_steps=cfg.compute.hardware.get(
                "gradient_accumulation", 1
            ),
            num_train_epochs=(
                cfg.model.specialise.epochs
                if cfg.model.specialise.epochs != "auto"
                else 3
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
            fp16=(torch_dtype == torch.float16),
            bf16=(torch_dtype == torch.bfloat16),
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds[cfg.data.source.split_map.train],
            eval_dataset=ds[cfg.data.source.split_map.valid],
            data_collator=data_collator,
            tokenizer=tok,
        )
        set_determinism(seeds[0], cfg.repro.cudnn_benchmark, cfg.repro.allow_tf32)

        run = wandb_log.start(
            cfg, run_name=f"S1-{cfg.model.name}", tags=["stage1", "specialise"]
        )

        # The Hugging Face Trainer will automatically log to W&B if it's installed
        # and configured (e.g., via environment variables), as `report_to` is not
        # set to an empty list. The `wandb_log.start` helper is assumed to handle
        # this setup.

        trainer.train()

        wandb_log.finish()

        ckpt_dir = paths.checkpoints / f"{model_name}_stage1"
        (ckpt_dir / "done.txt").write_text("ok\n")
        log.info(f"[stage1] Done. Checkpoint at: {ckpt_dir}")

    _run()


if __name__ == "__main__":
    main()
