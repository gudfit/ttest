from __future__ import annotations
from typing import List, Tuple, Iterable
import torch
import torch.nn.functional as F


@torch.no_grad()
def pll_surprisal_scores(
    token_ids: torch.LongTensor, model, tokenizer, mask_token_id: int
) -> torch.Tensor:
    """
    Pseudo-log-likelihood surprisal per token for MLMs:
    For each position i, mask it and compute -log2 p(x_i | X\i).
    Returns tensor [seq_len] in bits (log2).
    """
    device = token_ids.device
    seq_len = token_ids.size(0)
    scores = torch.empty(seq_len, device=device, dtype=torch.float32)
    for i in range(seq_len):
        masked = token_ids.clone()
        masked[i] = mask_token_id
        outputs = model(input_ids=masked.unsqueeze(0))
        logits = outputs.logits[0, i]  # [V]
        logp = F.log_softmax(logits, dim=-1)
        tok = token_ids[i].item()
        nats = -logp[tok]
        bits = nats / torch.log(torch.tensor(2.0, device=device))
        scores[i] = bits
    return scores


def topk_global_mask(
    surprisal_bits: torch.Tensor, keep_fraction: float
) -> torch.Tensor:
    """
    Keep the lowest surprisal tokens (most predictable) masked out.
    Return boolean keep_mask (True=keep/original, False=mask).
    """
    seq_len = surprisal_bits.numel()
    k_keep = int(round(keep_fraction * seq_len))
    order = torch.argsort(surprisal_bits, dim=0)  # low→high
    mask_flags = torch.ones(seq_len, dtype=torch.bool, device=surprisal_bits.device)
    to_mask = order[: seq_len - k_keep]
    mask_flags[to_mask] = False
    return mask_flags


def entropy_equalisation_mask(
    surprisal_bits: torch.Tensor, keep_fraction: float, window: int = 64
) -> torch.Tensor:
    """
    Windowed 'water-filling': allocate masking within windows to equalise local information rate.
    """
    seq_len = surprisal_bits.numel()
    keep = torch.zeros(seq_len, dtype=torch.bool, device=surprisal_bits.device)
    for start in range(0, seq_len, window):
        end = min(seq_len, start + window)
        w = surprisal_bits[start:end]
        k_keep = int(round(keep_fraction * (end - start)))
        order = torch.argsort(w, dim=0, descending=True)
        idx = order[:k_keep] + start
        keep[idx] = True
    return keep
