# lldc/models/vq/vq_trainer.py

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import inspect
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from lldc.models.vq.vq_bottleneck import VQBottleneckWrapper
from tqdm.auto import tqdm

class _IndexGRULM(nn.Module):
    def __init__(self, K: int, hidden: int = 512, layers: int = 1):
        super().__init__()
        self.K = int(K)
        self.emb = nn.Embedding(self.K, hidden)
        self.rnn = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, self.K)
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        h = self.emb(x)
        y, _ = self.rnn(h)
        return self.out(y)

@torch.no_grad()
def cross_entropy_bits_index_stream(model: _IndexGRULM, idx: List[int]) -> float:
    if not idx or len(idx) <= 1:
        return 0.0
    device = next(model.parameters()).device
    x = torch.tensor(idx[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(idx[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x)
    logp = torch.log_softmax(logits, dim=-1)
    nll = nn.functional.nll_loss(logp.view(-1, logp.size(-1)), y.view(-1), reduction="mean")
    nll_bits = float(nll.item() / math.log(2.0))
    return nll_bits * (len(idx) - 1)

def train_index_lm(
    sequences: List[List[int]],
    K: int,
    hidden: int = 512,
    layers: int = 1,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> _IndexGRULM:
    model = _IndexGRULM(K=K, hidden=hidden, layers=layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()
    def _examples():
        for seq in sequences:
            if not seq or len(seq) <= 1:
                continue
            yield torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)
    def _collate(batch):
        if not batch:
            x = torch.zeros((1, 1), dtype=torch.long)
            y = torch.zeros((1, 1), dtype=torch.long)
            return x, y
        max_len = max(b[0].numel() for b in batch)
        xs, ys = [], []
        for x, y in batch:
            if x.numel() == 0 or y.numel() == 0:
                continue
            pad_x = torch.full((max_len,), 0, dtype=torch.long)
            pad_y = torch.full((max_len,), 0, dtype=torch.long)
            pad_x[: x.numel()] = x
            pad_y[: y.numel()] = y
            xs.append(pad_x)
            ys.append(pad_y)
        if not xs:
            x = torch.zeros((1, 1), dtype=torch.long)
            y = torch.zeros((1, 1), dtype=torch.long)
            return x, y
        xs = torch.stack(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        return xs, ys
    loader = DataLoader(list(_examples()), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(max(1, int(epochs))):
        pbar = tqdm(loader, desc=f"Training Index LM Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            logp = nn.functional.log_softmax(logits, dim=-1)
            loss = nn.functional.nll_loss(logp.view(-1, logp.size(-1)), y.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
    model.eval()
    return model

def _tok_from_pretrained(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None and tok.eos_token:
        tok.pad_token = tok.eos_token
    return tok

def _filter_nonempty_ids(ds):
    return ds.filter(lambda ex: isinstance(ex.get("input_ids", None), list) and len(ex["input_ids"]) > 0)

@dataclass
class _TrainCfg:
    base_model_name: str
    dataset_name: str
    dataset_config: Optional[str]
    text_field: str
    max_length: int
    layer_after: int
    codebook_size: int
    lr: float
    epochs: int
    beta: float

def _tokenize_map_fn(tok, text_field: str, max_length: int):
    def _fn(batch):
        texts = batch[text_field]
        enc = tok(
            texts,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask", None)}
    return _fn

def _collate_padded(tok):
    collator = DataCollatorWithPadding(tokenizer=tok, padding=True, return_tensors="pt")
    def _fn(features: List[Dict]) -> Dict[str, torch.Tensor]:
        feats = [f for f in features if "input_ids" in f and len(f["input_ids"]) > 0]
        if not feats:
            feats = [{"input_ids": [tok.eos_token_id or 0]}]
        return collator(feats)
    return _fn

def _build_vq_wrapper(base_model, layer_after: int, codebook_size: int, beta: float) -> VQBottleneckWrapper:
    return VQBottleneckWrapper(lm=base_model, layer_after=layer_after, codebook_size=codebook_size, beta=beta)

def train_vq_joint(
    base_model_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    text_field: str,
    max_length: int,
    layer_after: int,
    codebook_size: int,
    lr: float = 5e-5,
    epochs: int = 2,
    beta: float = 0.25,
    max_train_samples: Optional[int] = None,
    **_: Any,
) -> Tuple[VQBottleneckWrapper, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_tok = _tok_from_pretrained(base_model_name)
    ds = load_dataset(dataset_name, dataset_config)
    train_split = ds["train"]
    if max_train_samples is not None and len(train_split) > max_train_samples:
        train_split = train_split.select(range(max_train_samples))
    map_fn = _tokenize_map_fn(base_tok, text_field, max_length)
    train_tok = train_split.map(map_fn, batched=True, remove_columns=[text_field])
    train_tok = _filter_nonempty_ids(train_tok)
    if len(train_tok) == 0:
        raise RuntimeError("After tokenization, no training examples remained (all were empty). Please check your dataset/text_field.")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model.to(device)
    base_model.train()
    model = _build_vq_wrapper(
        base_model=base_model,
        layer_after=int(layer_after),
        codebook_size=int(codebook_size),
        beta=float(beta),
    ).to(device)
    loader = DataLoader(
        train_tok,
        batch_size=8,
        shuffle=True,
        collate_fn=_collate_padded(base_tok),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=float(lr))
    total_steps = max(1, len(loader) * max(1, int(epochs)))
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
    )
    model.train()
    for epoch in range(max(1, int(epochs))):
        pbar = tqdm(loader, desc=f"Training VQ Model Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch.get("attention_mask", None)
            labels = input_ids.clone()
            if attn is not None:
                labels[attn.to(labels.device) == 0] = -100
            out = model(input_ids=input_ids, attention_mask=attn.to(device) if attn is not None else None, labels=labels)
            loss = out['loss']
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            pbar.set_postfix(loss=loss.item())
    model.eval()
    return model, base_tok

@torch.no_grad()
def encode_indices(model: VQBottleneckWrapper, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if input_ids is None or input_ids.numel() == 0 or input_ids.shape[-1] == 0:
        dev = next(model.parameters()).device
        return torch.empty((0,), dtype=torch.long, device=dev), None
    if hasattr(model, "encode_to_indices"):
        idx = model.encode_to_indices(input_ids)
    elif hasattr(model, "encode_indices"):
        idx = model.encode_indices(input_ids)
    else:
        out = model(input_ids=input_ids)
        if 'indices' in out:
            idx = out['indices']
        else:
            dev = next(model.parameters()).device
            return torch.empty((0,), dtype=torch.long, device=dev), None
    if isinstance(idx, (list, tuple)):
        idx = torch.tensor(idx, dtype=torch.long, device=next(model.parameters()).device)
    idx = idx.squeeze()
    if idx.dim() == 0:
        idx = idx.unsqueeze(0)
    return idx, None
