"""Tiny re-implementation of the Qwen3 backbone used by masked-diffusion LMs (A2D-Qwen3, LLaDA-style).

We do NOT execute the repo's remote code: config.json + safetensors are read directly.
`bidirectional=True` (default) = diffusion LM (no causal mask); `bidirectional=False` = plain Qwen3 (used only to
check this implementation against transformers' native Qwen3 in tests).

forward() returns every layer's residual stream so activation trajectories and a logit lens come for free:
    hs[0]   = token embeddings
    hs[l]   = output of decoder layer l (1..L), NOT yet passed through the final RMSNorm
    lens(h) = lm_head(final_norm(h))   -> "logit lens" at any layer
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F


def _rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def _rot(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


class DiffusionQwen3:
    def __init__(self, repo="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", dtype=torch.float32):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        self.cfg = json.load(open(hf_hub_download(repo, "config.json")))
        c = self.cfg
        self.L, self.nh, self.nkv, self.hd = c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.eps, self.theta, self.dtype = c["rms_norm_eps"], c["rope_theta"], dtype
        sd = {k: v.to(dtype) for k, v in load_file(hf_hub_download(repo, "model.safetensors")).items()}
        self.sd = sd
        self.embed = sd["model.embed_tokens.weight"]
        self.head = sd.get("lm_head.weight", self.embed)
        self.norm_w = sd["model.norm.weight"]

    # ------------------------------------------------------------------
    def _rope(self, T, device):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=device).float() / self.hd))
        f = torch.outer(torch.arange(T, device=device).float(), inv)
        e = torch.cat([f, f], -1)
        return e.cos().to(self.dtype), e.sin().to(self.dtype)

    @torch.no_grad()
    def forward(self, ids, bidirectional=True, collect=True):
        """ids [B,T] -> (logits [B,T,V], hidden list of L+1 tensors [B,T,D])."""
        B, T = ids.shape
        sd, p = self.sd, "model.layers.%d."
        x = self.embed[ids]
        cos, sin = self._rope(T, ids.device)
        hs = [x] if collect else []
        for l in range(self.L):
            q_ = p % l
            h = _rms(x, sd[q_ + "input_layernorm.weight"], self.eps)
            q = (h @ sd[q_ + "self_attn.q_proj.weight"].T).view(B, T, self.nh, self.hd)
            k = (h @ sd[q_ + "self_attn.k_proj.weight"].T).view(B, T, self.nkv, self.hd)
            v = (h @ sd[q_ + "self_attn.v_proj.weight"].T).view(B, T, self.nkv, self.hd)
            q = _rms(q, sd[q_ + "self_attn.q_norm.weight"], self.eps).transpose(1, 2)
            k = _rms(k, sd[q_ + "self_attn.k_norm.weight"], self.eps).transpose(1, 2)
            v = v.transpose(1, 2)
            q = q * cos + _rot(q) * sin
            k = k * cos + _rot(k) * sin
            rep = self.nh // self.nkv
            a = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1),
                                               is_causal=not bidirectional)
            x = x + a.transpose(1, 2).reshape(B, T, -1) @ sd[q_ + "self_attn.o_proj.weight"].T
            h = _rms(x, sd[q_ + "post_attention_layernorm.weight"], self.eps)
            m = F.silu(h @ sd[q_ + "mlp.gate_proj.weight"].T) * (h @ sd[q_ + "mlp.up_proj.weight"].T)
            x = x + m @ sd[q_ + "mlp.down_proj.weight"].T
            if collect:
                hs.append(x)
        logits = _rms(x, self.norm_w, self.eps) @ self.head.T
        return logits, hs

    @torch.no_grad()
    def lens(self, h):
        """logit lens: final RMSNorm + unembedding applied to any intermediate residual stream."""
        return _rms(h, self.norm_w, self.eps) @ self.head.T
