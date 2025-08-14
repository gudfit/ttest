# lldc/metrics/crumpled_paper.py

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class Oracle:
    name: str = "gpt2-large"
    device: str = "cuda"

    def __post_init__(self):
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.m = AutoModelForCausalLM.from_pretrained(self.name).to(self.device).eval()
        self.vocab_size = int(getattr(self.m.config, "vocab_size", len(self.tok)))

    @torch.no_grad()
    def surprisal_bits(self, text: str) -> Tuple[List[int], List[float]]:
        enc = self.tok(text, add_special_tokens=False, return_tensors="pt").to(
            self.device
        )
        ids = enc["input_ids"][0]
        logits = self.m(input_ids=ids.unsqueeze(0)).logits[0]
        if ids.size(0) < 2:
            return ids.tolist(), []
        logp = torch.log_softmax(logits[:-1], dim=-1)
        true_next = ids[1:].unsqueeze(-1)
        nats = -logp.gather(-1, true_next).squeeze(-1)
        bits = (nats / math.log(2.0)).tolist()
        return ids.tolist(), [float(x) for x in bits]


def _needleman_wunsch_cost(
    a_s: List[float], b_s: List[float], gap_cost: float
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Align two surprisal sequences with global NW:
      - match/mismatch cost = |a_s[i] - b_s[j]|
      - gap cost = gap_cost per gap symbol
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
            if c_sub <= c_del and c_sub <= c_ins:
                dp[i][j] = c_sub
                bt[i][j] = (i - 1, j - 1)
            elif c_del <= c_ins:
                dp[i][j] = c_del
                bt[i][j] = (i - 1, j)
            else:
                dp[i][j] = c_ins
                bt[i][j] = (i, j - 1)

    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj = bt[i][j]
        if pi == i - 1 and pj == j - 1:
            path.append((i - 1, j - 1))
            i, j = pi, pj
        elif pi == i - 1 and pj == j:
            path.append((i - 1, -1))
            i, j = pi, pj
        else:
            path.append((-1, j - 1))
            i, j = pi, pj
    path.reverse()
    return dp[n][m], path


def tcm_pcm_from_texts(orig: str, recon: str, oracle: Oracle) -> Dict[str, float]:
    _, s0 = oracle.surprisal_bits(orig)
    _, s1 = oracle.surprisal_bits(recon)
    return tcm_pcm_from_surprisal(s0, s1, vocab_size=oracle.vocab_size)


def tcm_pcm_from_surprisal(
    s0: List[float], s1: List[float], vocab_size: int
) -> Dict[str, float]:
    gap = math.log2(max(2, vocab_size))
    _, path = _needleman_wunsch_cost(s0, s1, gap_cost=gap)
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
