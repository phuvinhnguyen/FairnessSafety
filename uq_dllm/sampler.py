"""LLaDA-style masked-diffusion sampler (semi-autoregressive blocks, low-confidence remasking, greedy)
that records the full activation trajectory: hidden states of every layer at every denoising step.

Time axis: s = 0..S-1 are the S denoising forward passes (input still contains <|mask|>), s = S is one extra
forward pass on the final, fully unmasked answer (the "clean" reference used for 'change relative to final answer').
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

REPO = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"


def load_tokenizer(repo=REPO):
    """Tokenizer without any remote code: tokenizer.json + chat_template.jinja only."""
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(repo, "tokenizer.json"),
                                  mask_token="<|mask|>", eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = open(hf_hub_download(repo, "chat_template.jinja")).read()
    return tok


def encode_prompt(tok, user, system="You are a helpful AI assistant."):
    text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def generate(model, tok, prompt_ids, G=128, steps=64, block=32):
    """Returns dict(tokens, hidden[S+1,L+1,G,D] fp32, per-step top1 prob / entropy / argmax, commit stats)."""
    mask_id, eos_id = tok.mask_token_id, tok.eos_token_id
    P = len(prompt_ids)
    x = torch.tensor([list(prompt_ids) + [mask_id] * G])
    nb = G // block
    spb = steps // nb
    hid, top1, ent, amax = [], [], [], []
    c_conf, c_ent = np.full(G, np.nan), np.full(G, np.nan)
    c_step = np.full(G, -1)
    step = 0
    for b in range(nb):
        bs, be = P + b * block, P + (b + 1) * block
        n_mask = int((x[0, bs:be] == mask_id).sum())
        ntr = [n_mask // spb + (1 if i < n_mask % spb else 0) for i in range(spb)]
        for i in range(spb):
            logits, hs = model.forward(x, bidirectional=True, collect=True)
            hid.append(torch.stack([h[0, P:] for h in hs]))            # [L+1, G, D]
            p = F.softmax(logits[0, P:].float(), -1)                    # [G, V]
            mp, am = p.max(-1)
            e = -(p * torch.log(p + 1e-12)).sum(-1)
            top1.append(mp.numpy()); ent.append(e.numpy()); amax.append(am.numpy())
            masked = (x[0, P:] == mask_id)
            blk = torch.zeros(G, dtype=torch.bool); blk[bs - P:be - P] = True
            cand = masked & blk
            conf = torch.where(cand, mp, torch.tensor(-1.0))
            k = ntr[i]
            if k > 0:
                sel = torch.topk(conf, k).indices
                x[0, P + sel] = am[sel]
                c_conf[sel.numpy()], c_ent[sel.numpy()], c_step[sel.numpy()] = mp[sel].numpy(), e[sel].numpy(), step
            step += 1
    # final clean pass on the fully decoded answer
    logits, hs = model.forward(x, bidirectional=True, collect=True)
    hid.append(torch.stack([h[0, P:] for h in hs]))
    p = F.softmax(logits[0, P:].float(), -1)
    y = x[0, P:]
    fin_p = p.gather(-1, y[:, None])[:, 0].numpy()                      # p_final(y_t | rest of clean answer)
    fin_top1 = p.max(-1).values.numpy()
    fin_ent = (-(p * torch.log(p + 1e-12)).sum(-1)).numpy()
    toks = y.tolist()
    n_content = toks.index(eos_id) if eos_id in toks else G
    return dict(tokens=toks, n_content=max(1, n_content), hidden=torch.stack(hid), S=steps,
                top1=np.stack(top1), ent=np.stack(ent), argmax=np.stack(amax),
                c_conf=c_conf, c_ent=c_ent, c_step=c_step, fin_p=fin_p, fin_top1=fin_top1, fin_ent=fin_ent,
                P=P, G=G)
