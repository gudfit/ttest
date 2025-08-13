from __future__ import annotations
from typing import Any
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from lldc.utils.logging import setup_logging
from lldc.utils.hydra_utils import resolve_auto
from lldc.utils.paths import Paths
from lldc.models.specialization import structured_prune


def _arch_from_name(name: str) -> str:
    n = name.lower()
    return "ar" if "gpt2" in n else "mlm"


def main():
    import hydra

    @hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
    def _run(cfg: Any) -> None:
        log = setup_logging()
        cfg = resolve_auto(cfg)
        paths = Paths().ensure()

        model_name = cfg.model.pretrained_name
        level = float(getattr(cfg, "prune_level", 0.1))
        arch = _arch_from_name(model_name)
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None and tok.eos_token:
            tok.pad_token = tok.eos_token

        if arch == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        dropped = structured_prune(
            model,
            level=level,
            drop_heads=cfg.experiment.pruning.structured.drop_attention_heads,
            drop_ffn=cfg.experiment.pruning.structured.drop_ffn_channels,
        )
        log.info(f"[prune] {model_name} @ level={level} → dropped={dropped}")

        # brief recovery finetune (small budget)
        ds = load_dataset(cfg.data.source.hf_dataset, cfg.data.source.hf_config)
        text_field = cfg.data.processing.text_field

        def tok_fn(b):
            return tok(
                b[text_field],
                truncation=True,
                max_length=cfg.data.processing.max_length,
            )

        ds = ds.map(tok_fn, batched=True, remove_columns=[text_field])
        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=(arch == "mlm"))
        args = TrainingArguments(
            output_dir=str(paths.checkpoints / f"{model_name}_pruned_{level}"),
            per_device_train_batch_size=2,
            num_train_epochs=1,
            learning_rate=5e-5,
            bf16=torch.cuda.is_available(),
            evaluation_strategy="no",
            save_strategy="no",
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=collator,
            train_dataset=ds["train"].select(range(min(4000, len(ds["train"])))),
        )
        trainer.train()

        outdir = paths.checkpoints / f"{model_name}_pruned_{level}"
        outdir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(outdir)
        tok.save_pretrained(outdir)
        (outdir / "READY").write_text("ok\n")
        log.info(f"[prune] Saved pruned+recovered checkpoint → {outdir}")

    _run()


if __name__ == "__main__":
    main()
