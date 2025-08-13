from __future__ import annotations
from typing import Iterable, List, Sequence
import numpy as np

try:
    import constriction
except Exception:
    constriction = None


def encode_with_probs(
    symbols: Sequence[int], probs: Sequence[Sequence[float]]
) -> bytes:
    """
    Arithmetic-code `symbols` given per-step categorical probabilities `probs`.
    probs[t] is a length-V vector summing to 1.
    """
    if constriction is None:
        return bytes(symbols)
    coder = constriction.stream.queue_bit_coder.QueueBitCoder()
    for s, p in zip(symbols, probs):
        p_arr = np.asarray(p, dtype=np.float64)
        p_arr = p_arr / p_arr.sum()
        cdf = np.cumsum(p_arr)
        coder.encode_symbol_using_cdf(s, cdf)
    return coder.get_compressed()


def decode_with_probs(payload: bytes, probs: Sequence[Sequence[float]]) -> List[int]:
    if constriction is None:
        return list(payload)
    decoder = constriction.stream.queue_bit_coder.QueueBitCoder(payload)
    out: List[int] = []
    for p in probs:
        p_arr = np.asarray(p, dtype=np.float64)
        p_arr = p_arr / p_arr.sum()
        cdf = np.cumsum(p_arr)
        s = decoder.decode_symbol_using_cdf(cdf)
        out.append(int(s))
    return out


def payload_num_bits(payload: bytes) -> int:
    return len(payload) * 8
