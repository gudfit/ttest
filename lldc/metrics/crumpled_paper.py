from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch


@dataclass
class Oracle:
    name: str = "gpt2-large"
    device: str = "cuda"

    def __post_init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.m = AutoModelForCausalLM.from_pretrained(self.name).to(self.device).eval()
        self.vocab_size = int(getattr(self.m.config, "vocab_size", len(self.tok)))

    @torch.no_grad()
    def surprisal_bits(self, text: str) -> Tuple[List[int], List[float]]:
        enc = self.tok(text, add_special_tokens=False, return_tensors="pt").to(
            self.device
        )
        ids = enc["input_ids"][0]  # [L]
        # teacher-force causal Xent; per-token surprisal in bits
        out = self.m(input_ids=ids.unsqueeze(0), labels=ids.unsqueeze(0))
        # out.loss is mean; we want tokenwise:
        logits = self.m(input_ids=ids.unsqueeze(0)).logits[0]  # [L,V]
        probs = torch.softmax(logits, dim=-1)
        ps = probs[torch.arange(ids.size(0)), ids].clamp_min(1e-12)  # [L]
        s_bits = (-torch.log2(ps)).tolist()
        return ids.tolist(), [float(x) for x in s_bits]


def _needleman_wunsch_cost(
    a_s: List[float], b_s: List[float], gap_cost: float
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Align two surprisal sequences with global NW:
      - match/mismatch cost = |a_s[i] - b_s[j]|
      - gap cost = gap_cost per gap symbol
    Returns (total_cost, path) where path is list of (i or -1, j or -1).
    """
    n, m = len(a_s), len(b_s)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[(0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        bt[i][0] = (i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j * gap_cost
        bt[0][j] = (0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c_sub = dp[i - 1][j - 1] + abs(a_s[i - 1] - b_s[j - 1])
            c_del = dp[i - 1][j] + gap_cost
            c_ins = dp[i][j - 1] + gap_cost
            best = min(
                (c_sub, (i - 1, j - 1)),
                (c_del, (i - 1, j)),
                (c_ins, (i, j - 1)),
                key=lambda z: z[0],
            )
            dp[i][j], bt[i][j] = best

    # backtrack to path of (i index or -1 for gap, j index or -1)
    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj = bt[i][j]
        if pi == i - 1 and pj == j - 1:  # sub/match
            path.append((i - 1, j - 1))
            i, j = pi, pj
        elif pi == i - 1 and pj == j:  # deletion in b  (a aligned to gap)
            path.append((i - 1, -1))
            i, j = pi, pj
        else:  # insertion in b (gap aligned to b)
            path.append((-1, j - 1))
            i, j = pi, pj
    path.reverse()
    return dp[n][m], path


def tcm_pcm_from_texts(orig: str, recon: str, oracle: Oracle) -> Dict[str, float]:
    # single measurement tokenizer: oracle tokenizer for both strings
    _, s0 = oracle.surprisal_bits(orig)
    _, s1 = oracle.surprisal_bits(recon)
    return tcm_pcm_from_surprisal(s0, s1, vocab_size=oracle.vocab_size)


def tcm_pcm_from_surprisal(
    s0: List[float], s1: List[float], vocab_size: int
) -> Dict[str, float]:
    gap = math.log2(max(2, vocab_size))
    _, path = _needleman_wunsch_cost(s0, s1, gap_cost=gap)
    # accumulate per-column cost along alignment path
    tcm = 0.0
    pcm = 0.0
    for i, j in path:
        if i == -1 or j == -1:
            c = gap
        else:
            c = abs(s0[i] - s1[j])
        tcm += c
        pcm = max(pcm, c)
    return {"tcm_bits": float(tcm), "pcm_bits": float(pcm)}
