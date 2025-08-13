from __future__ import annotations
from typing import List, Dict
import math, torch, numpy as np


def ppl_causal(model, tok, texts: List[str], device="cuda") -> float:
    model.to(device).eval()
    losses = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt").to(device)
            out = model(**enc, labels=enc["input_ids"])
            losses.append(out.loss.item())
    H_bits = float(np.mean(losses)) / math.log(2)
    return float(2**H_bits)


def pppl_mlm(model, tok, texts: List[str], device="cuda") -> float:
    model.to(device).eval()
    log2ps = []
    mask_id = tok.mask_token_id
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").to(device)["input_ids"][0]
            L = ids.size(0)
            sum_log2p = 0.0
            for i in range(L):
                masked = ids.clone()
                masked[i] = mask_id
                logits = model(input_ids=masked.unsqueeze(0)).logits[0, i]
                p = torch.softmax(logits, dim=-1)[ids[i]].item()
                sum_log2p += math.log(p + 1e-12, 2)
            log2ps.append(sum_log2p / max(1, L))
    H_bits = -float(np.mean(log2ps))
    return float(2**H_bits)


def compute_ppl(model, tok, texts: List[str], device="cuda") -> Dict:
    if hasattr(model, "generate"):
        return {"ppl": ppl_causal(model, tok, texts, device)}
    else:
        return {"pppl": pppl_mlm(model, tok, texts, device)}
