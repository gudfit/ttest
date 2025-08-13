from __future__ import annotations
import math
from lldc.metrics.crumpled_paper import Oracle, tcm_pcm_from_surprisal


def test_gap_cost_single_token():
    orc = Oracle("sshleifer/tiny-gpt2")
    gap = math.log2(max(2, orc.vocab_size))
    # empty vs one-token surprisal → cost equals gap
    _, s1 = orc.surprisal_bits("Hello")
    m = tcm_pcm_from_surprisal([], s1, vocab_size=orc.vocab_size)
    assert abs(m["tcm_bits"] - gap) < 1e-4
    assert abs(m["pcm_bits"] - gap) < 1e-4
