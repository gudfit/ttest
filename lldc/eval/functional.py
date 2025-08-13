from __future__ import annotations
from typing import List, Dict
import numpy as np, torch


def evaluate_superglue_zero_shot(
    model, tok, tasks: List[str], n: int = 100, device="cuda"
) -> Dict:
    from datasets import load_dataset

    model.to(device).eval()
    specs = {
        "rte": (
            "super_glue",
            "rte",
            ("premise", "hypothesis"),
            ["entailment", "not_entailment"],
        ),
        "cb": (
            "super_glue",
            "cb",
            ("premise", "hypothesis"),
            ["entailment", "contradiction", "neutral"],
        ),
        "boolq": ("super_glue", "boolq", ("passage", "question"), ["yes", "no"]),
    }
    import evaluate

    macro = evaluate.load("f1", "multiclass")
    out_task, f1s = {}, []
    with torch.no_grad():
        for t in tasks:
            if t not in specs:
                continue
            name, subset, fields, labels = specs[t]
            ds = load_dataset(name, subset)["validation"]
            k = min(n, len(ds))
            preds, refs = [], []
            for i in range(k):
                a, b = ds[i][fields[0]], ds[i][fields[1]]
                prompt = f"{a}\n{b}\nAnswer with one of: {', '.join(labels)}."
                enc = tok(prompt, return_tensors="pt").to(device)
                if hasattr(model, "generate"):
                    out_ids = model.generate(**enc, max_new_tokens=8)
                    gen = tok.decode(
                        out_ids[0][enc["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    ).lower()
                else:
                    scores = []
                    for lab in labels:
                        inp = tok(prompt + " " + lab, return_tensors="pt").to(device)
                        loss = model(**inp, labels=inp["input_ids"]).loss.item()
                        scores.append(-loss)
                    gen = labels[int(np.argmax(scores))]
                if t == "boolq":
                    pred = 1 if "yes" in gen else 0
                    ref = int(ds[i]["label"])
                elif t == "rte":
                    pred = 1 if "entail" in gen else 0
                    ref = int(ds[i]["label"])
                else:  # cb
                    if "entail" in gen:
                        pred = 0
                    elif "contrad" in gen:
                        pred = 1
                    else:
                        pred = 2
                    ref = int(ds[i]["label"])
                preds.append(pred)
                refs.append(ref)
            f1 = macro.compute(predictions=preds, references=refs, average="macro")[
                "f1"
            ]
            out_task[t] = float(f1)
            f1s.append(f1)
    return {"per_task_f1": out_task, "macro_f1": float(np.mean(f1s) if f1s else 0.0)}
