# lldc/baselines/kenlm_subsample.py

from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
from collections import Counter
import os, subprocess
import kenlm, math
def train_kenlm(sentences: List[str], order: int, workdir: str, pruning: str = "0 0 1") -> Tuple[str, str]:
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    corpus = wd / "train.txt"
    corpus.write_text("\n".join(sentences), encoding="utf-8")
    bin_dir = Path(os.environ.get("KENLM_BIN", "/usr/local/bin"))
    lmplz = bin_dir / "lmplz"
    build_binary = bin_dir / "build_binary"
    if not lmplz.exists() or not build_binary.exists():
        raise RuntimeError(
            "KenLM binaries not found. Please install KenLM and set KENLM_BIN to the "
            "directory containing 'lmplz' and 'build_binary' (e.g., export KENLM_BIN=/path/to/kenlm/build/bin)."
        )
    arpa = wd / f"lm{order}.arpa"
    binary = wd / f"lm_{order}.binary"
    mem = os.environ.get("KENLM_MEMORY", "70%")
    tmp = wd / "tmp"
    tmp.mkdir(exist_ok=True)
    cmd_lmplz = (
        f'"{lmplz}" -o {order} --discount_fallback '
        f'--prune {pruning} --temp_prefix "{tmp}" --memory {mem} '
        f'< "{corpus}" > "{arpa}"'
    )
    try:
        subprocess.check_call(cmd_lmplz, shell=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"KenLM lmplz failed (order={order}). Hint: ensure enough RAM (KENLM_MEMORY) "
            f"and that your binaries are compatible. Original error: {e}"
        ) from e
    cmd_bin = f'"{build_binary}" -a 255 -q 8 -v trie "{arpa}" "{binary}"'
    try:
        subprocess.check_call(cmd_bin, shell=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"KenLM build_binary failed. Check KENLM_BIN and permissions. Original error: {e}"
        ) from e
    return str(arpa), str(binary)
def _load_kenlm(arpa_path: str, binary_path: str | None = None):
    if binary_path and Path(binary_path).exists():
        return kenlm.Model(binary_path)
    return kenlm.Model(arpa_path)
def _build_vocab(docs: List[str], max_vocab: int = 50000) -> List[str]:
    cnt = Counter()
    for t in docs:
        cnt.update(t.split())
    return [w for w, _ in cnt.most_common(max_vocab)]
def _uniform_subsample_words(words: List[str], N: int) -> Tuple[List[str], List[int]]:
    kept = words[::N]
    idxs = list(range(0, len(words), N))
    return kept, idxs
def _state_from_context(model, ctx: List[str]):
    st = kenlm.State()
    model.BeginSentenceWrite(st)
    for w in ctx[-4:]:
        nxt = kenlm.State()
        model.BaseScore(st, w, nxt)
        st = nxt
    return st
def _viterbi_fill(
    model,
    ctx: List[str],
    end_word: str,
    gap_len: int,
    vocab: List[str],
    beam_size: int = 16,
    cand_per_step: int = 200,
) -> List[str]:
    if gap_len <= 0:
        return []
    cand = vocab[:cand_per_step]
    beams = [(_state_from_context(model, ctx), [], 0.0)]
    for _ in range(gap_len):
        nxt = []
        for st, seq, sc in beams:
            for w in cand:
                st2 = kenlm.State()
                inc = model.BaseScore(st, w, st2)
                nxt.append((st2, seq + [w], sc + inc))
        nxt.sort(key=lambda x: x[2], reverse=True)
        beams = nxt[:beam_size]
    best = None
    for st, seq, sc in beams:
        st2 = kenlm.State()
        inc = model.BaseScore(st, end_word, st2)
        if best is None or (sc + inc) > best[2]:
            best = (st2, seq, sc + inc)
    return best[1] if best else []
def subsample_and_reconstruct_kenlm5(
    test_texts: List[str],
    train_texts: List[str],
    rates: List[int],
    workdir: str,
    beam_size: int = 16,
    cand_per_step: int = 200,
    max_vocab: int = 50000,
) -> List[Dict]:
    arpa, binary = train_kenlm(train_texts, order=5, workdir=workdir)
    lm = _load_kenlm(arpa, binary)
    vocab = _build_vocab(train_texts, max_vocab=max_vocab)
    outputs = []
    for N in rates:
        recons, payloads = [], []
        for t in test_texts:
            words = t.split()
            kept, idxs = _uniform_subsample_words(words, N)
            payloads.append(" ".join(kept))
            out: List[str] = []
            for j, i in enumerate(idxs):
                out.append(words[i])
                if j + 1 < len(idxs):
                    gap = max(0, idxs[j + 1] - i - 1)
                    if gap > 0:
                        fill = _viterbi_fill(
                            lm,
                            out[:],
                            words[idxs[j + 1]],
                            gap,
                            vocab,
                            beam_size=beam_size,
                            cand_per_step=cand_per_step,
                        )
                        out.extend(fill)
            recons.append(" ".join(out))
        outputs.append(
            {"rate_N": N, "reconstructions": recons, "subsamples_payload": payloads}
        )
    return outputs
def kenlm_ngram_bpc(
    train_texts: List[str], test_texts: List[str], workdir: str, order: int = 8, pruning: str = "0 0 1"
) -> float:
    arpa, binary = train_kenlm(train_texts, order=order, workdir=workdir, pruning=pruning)
    lm = _load_kenlm(arpa, binary)
    total_log10 = 0.0
    total_chars = 0
    for t in test_texts:
        total_log10 += lm.score(t, bos=True, eos=True)
        total_chars += len(t)
    bits = -total_log10 / math.log10(2.0)
    return bits / max(1, total_chars)
