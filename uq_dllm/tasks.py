"""Small task loaders + scorers: GSM8K (exact match), WMT14 de->en translation (sentence chrF), ARC-Easy (accuracy)."""
from __future__ import annotations

import re

import numpy as np

SETTINGS = {  # answer length G, denoising steps, semi-AR block size
    "gsm8k": dict(G=128, steps=64, block=32),
    "wmt": dict(G=64, steps=32, block=32),
    "arc": dict(G=16, steps=16, block=16),
}


def _pq(repo, path):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset"))


def load(name, n, seed=0, offset=0):
    rng = np.random.default_rng(seed)
    if name == "gsm8k":
        d = _pq("openai/gsm8k", "main/test-00000-of-00001.parquet")
        idx = rng.permutation(len(d))
        items = [dict(id=int(i), user=d.question[i] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
                      gold=float(d.answer[i].split("####")[-1].strip().replace(",", ""))) for i in idx]
    elif name == "wmt":
        d = _pq("wmt/wmt14", "de-en/test-00000-of-00001.parquet")
        idx = rng.permutation(len(d))
        items = []
        for i in idx:
            t = d.translation[i]
            if 8 <= len(t["de"].split()) <= 30:
                items.append(dict(id=int(i), user="Translate the following sentence from German to English. "
                                  "Reply with the English translation only.\n\n" + t["de"], gold=t["en"]))
    elif name == "arc":
        d = _pq("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet")
        idx = rng.permutation(len(d))
        items = []
        for i in idx:
            ch = d.choices[i]
            labels = list(ch["label"])
            if labels == ["A", "B", "C", "D"] and d.answerKey[i] in labels:
                opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, ch["text"]))
                items.append(dict(id=int(i), user=f"{d.question[i]}\n{opts}\nAnswer with the letter of the correct option only.",
                                  gold=d.answerKey[i]))
    else:
        raise ValueError(name)
    return items[offset:offset + n]


def _number(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    try:
        return float(nums[-1].replace(",", "")) if nums else None
    except ValueError:
        return None


def score(name, text, gold):
    """-> (score, label). label=None means 'continuous only' (translation: binarised later)."""
    text = text.strip()
    if name == "gsm8k":
        m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
        pred = _number(m[-1]) if m else _number(text)
        ok = pred is not None and abs(pred - gold) < 1e-4
        return float(ok), int(ok)
    if name == "wmt":
        from sacrebleu.metrics import CHRF
        text = text.split("\n\n")[0].strip()  # the model often appends an explanation after the translation
        return float(CHRF().sentence_score(text, [gold]).score), None
    if name == "arc":
        m = re.search(r"\b([ABCD])\b", text)
        ok = bool(m) and m.group(1) == gold
        return float(ok), int(ok)
    raise ValueError(name)
