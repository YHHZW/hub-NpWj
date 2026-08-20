"""
腾讯混元 Hy3 建模文件（教学/结构复现版）
========================================
Hy3 是腾讯混元团队开源的快慢思考融合 MoE 大模型，总参 295B / 激活 21B，
MTP 层 3.8B，Apache 2.0 协议开源，支持 256K 上下文。

本文聚焦其【结构特点】，忠实于官方 config.json（HYV3ForCausalLM / hy_v3）。

两大结构创新：
1. 大规模稀疏 MoE
   * 192 个路由专家 + 1 个共享专家，每 token 激活 top-8（仅激活约 7% 参数）
   * sigmoid 路由 + expert bias（moe_router_enable_expert_bias / use_sigmoid）
   * router_scaling_factor=2.826
2. 快慢思考融合（Hybrid Thinking）推理范式
   * reasoning_effort: no_think / low / high（对应系统1/系统2 的工程路由，非模型结构改动）
   * 通过特殊 token 触发不同推理深度

关键超参 (config.json)：
    hidden_size=4096, layers=80, heads=64, kv_heads=8 (GQA 8:1), head_dim=128
    vocab=120832, 256K context
    MoE: 192 experts, 8/tok, 1 shared, expert_hidden=1536, first 1 layer dense
    Dense FFN inter=13312
    RoPE theta=11,158,840, qk_norm=True, RMSNorm eps=1e-5
    1 层 MTP（num_nextn_predict_layers=1）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. 基础组件
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(orig_dtype)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _precompute_freqs(dim: int, max_seq_len: int, base: float) -> torch.Tensor:
    """预计算 RoPE 频率（复数形式）。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """对张量尾部维度应用 RoPE。支持 4D [b,s,h,d]。"""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, -1, 1, xc.shape[-1])
    out = torch.view_as_real(xc * freqs_cis).flatten(-2)
    return out.to(x.dtype)


# ============================================================================
# 2. GQA 注意力（qk_norm=True, GQA 8:1）
# ============================================================================

class Attention(nn.Module):
    """GQA 注意力：64 heads / 8 KV heads（8:1），qk_norm，head_dim=128。"""

    def __init__(self, hidden_size: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 rope_theta: float, max_seq_len: int, eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.q_proj = Linear(hidden_size, n_heads * head_dim)
        self.k_proj = Linear(hidden_size, n_kv_heads * head_dim)
        self.v_proj = Linear(hidden_size, n_kv_heads * head_dim)
        self.o_proj = Linear(n_heads * head_dim, hidden_size)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        self.softmax_scale = head_dim ** -0.5
        self.register_buffer("freqs_cis",
                             _precompute_freqs(head_dim, max_seq_len, rope_theta),
                             persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        b, s, _ = x.shape
        freqs = self.freqs_cis[:seq_len]
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rotary_emb(q, freqs)
        k = _apply_rotary_emb(k, freqs)
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)
        attn = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
        causal = torch.tril(torch.ones(s, seq_len, dtype=torch.bool, device=x.device))
        attn = attn.masked_fill(~causal.view(1, s, 1, seq_len), float("-inf"))
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.einsum("bsht,bthd->bshd", attn, v).reshape(b, s, -1)
        return self.o_proj(out)


# ============================================================================
# 3. SwiGLU FFN / MoE（192 experts, top-8, sigmoid, 1 shared）
# ============================================================================

class SwiGLU(nn.Module):
    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up


class Expert(nn.Module):
    """单个 MoE 专家 FFN：SwiGLU，中间维 1536。"""

    def __init__(self, hidden_size: int, inter_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, inter_size)
        self.up_proj = Linear(hidden_size, inter_size)
        self.down_proj = Linear(inter_size, hidden_size)
        self.act = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x), self.up_proj(x)))


class DenseFFN(nn.Module):
    """Dense FFN（前 1 层使用），中间维 13312。"""

    def __init__(self, hidden_size: int, inter_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, inter_size)
        self.up_proj = Linear(hidden_size, inter_size)
        self.down_proj = Linear(inter_size, hidden_size)
        self.act = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x), self.up_proj(x)))


class MoE(nn.Module):
    """
    Hy3 大规模稀疏 MoE：
    - 192 个路由专家，每 token 激活 top-8
    - 1 个共享专家（总是被激活）
    - sigmoid 路由 + expert bias（bias 只影响选择，不影响权重）
    - route_norm=True：top-8 权重 L1 归一化
    - router_scaling_factor=2.826
    """

    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int,
                 num_shared_experts: int, expert_inter_size: int,
                 router_scaling_factor: float, use_sigmoid: bool = True,
                 use_expert_bias: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.router_scaling_factor = router_scaling_factor
        self.use_sigmoid = use_sigmoid

        self.gate = Linear(hidden_size, num_experts)
        # expert bias：影响选择但不影响权重
        if use_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        else:
            self.register_buffer("expert_bias", None)

        self.experts = nn.ModuleList(
            [Expert(hidden_size, expert_inter_size) for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList(
            [Expert(hidden_size, expert_inter_size) for _ in range(num_shared_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        flat = x.reshape(-1, x.shape[-1])
        logits = self.gate(flat)
        if self.use_sigmoid:
            probs = torch.sigmoid(logits.float())
        else:
            probs = F.softmax(logits.float(), dim=-1)
        # 选择分数 = 路由分数 + expert bias
        select = probs.clone()
        if self.expert_bias is not None:
            select = select + self.expert_bias.float()
        topk_probs, topk_idx = select.topk(self.num_experts_per_tok, dim=-1)
        # 权重取自原始 sigmoid 概率（不含 bias），L1 归一化
        routed = probs.gather(1, topk_idx)
        routed = routed / (routed.sum(dim=-1, keepdim=True) + 1e-6)
        routed = routed * self.router_scaling_factor

        out = torch.zeros_like(flat, dtype=torch.float32)
        for e in range(self.num_experts):
            mask = (topk_idx == e)
            if mask.any():
                token_ids, slot = torch.where(mask)
                out[token_ids] += self.experts[e](flat[token_ids]) * routed[token_ids, slot, None]
        for sh in self.shared_experts:
            out += sh(flat).float()
        return out.reshape(b, s, -1).to(x.dtype)


# ============================================================================
# 4. Transformer Block + 完整模型 + MTP
# ============================================================================

class Block(nn.Module):
    def __init__(self, layer_id: int, cfg: "Hy3Config"):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention = Attention(
            cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
            cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings, cfg.rms_norm_eps)
        # 前 first_k_dense_replace 层用 Dense FFN，其余用 MoE
        if layer_id < cfg.first_k_dense_replace:
            self.mlp = DenseFFN(cfg.hidden_size, cfg.intermediate_size)
        else:
            self.mlp = MoE(
                cfg.hidden_size, cfg.num_experts, cfg.num_experts_per_tok,
                cfg.num_shared_experts, cfg.moe_intermediate_size,
                cfg.router_scaling_factor, cfg.moe_router_use_sigmoid,
                cfg.moe_router_enable_expert_bias)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        h = x + self.attention(self.input_layernorm(x), seq_len)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class MTPBlock(nn.Module):
    """多 token 预测（MTP）层：预测未来 token，提升推理吞吐与长程依赖建模。"""

    def __init__(self, cfg: "Hy3Config"):
        super().__init__()
        self.e_proj = Linear(cfg.hidden_size, cfg.hidden_size)
        self.h_proj = Linear(cfg.hidden_size, cfg.hidden_size)
        self.enorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.hnorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.block = Block(-1, cfg)   # 复用同一层结构（Dense FFN）

    def forward(self, x: torch.Tensor, embed_next: torch.Tensor, seq_len: int) -> torch.Tensor:
        x = self.hnorm(x)
        e = self.enorm(embed_next)
        h = self.e_proj(e) + self.h_proj(x)
        return self.block(h, seq_len)


@dataclass
class Hy3Config:
    """腾讯混元 Hy3 配置（对应 config.json）。"""
    hidden_size: int = 4096
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 120832
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    rope_theta: float = 11158840.0
    hidden_act: str = "silu"
    intermediate_size: int = 13312          # dense FFN 中间维
    first_k_dense_replace: int = 1          # 前 1 层 dense
    # MoE
    num_experts: int = 192
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    moe_intermediate_size: int = 1536       # expert_hidden_dim
    router_scaling_factor: float = 2.826
    moe_router_use_sigmoid: bool = True
    moe_router_enable_expert_bias: bool = True
    # MTP
    num_nextn_predict_layers: int = 1


class Hy3ForCausalLM(nn.Module):
    """完整 Hy3 模型（含 1 层 MTP）。"""

    def __init__(self, cfg: Hy3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(i, cfg) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mtp_layers = nn.ModuleList(
            [MTPBlock(cfg) for _ in range(cfg.num_nextn_predict_layers)])
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        b, s = input_ids.shape
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, s)
        h = self.norm(h)
        logits = self.lm_head(h)

        # MTP：预测未来 token 的 logits（投机解码用）
        mtp_logits = None
        if self.mtp_layers and s > 1:
            embed_next = self.embed_tokens(input_ids[:, 1:])   # 下一个 token 的 embedding
            h_mtp = self.mtp_layers[0](h[:, :-1], embed_next, s - 1)
            mtp_logits = self.lm_head(self.norm(h_mtp))
        return logits, mtp_logits


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    cfg = Hy3Config(num_hidden_layers=4)   # 小规模测试
    cfg.first_k_dense_replace = 1
    cfg.num_nextn_predict_layers = 1
    model = Hy3ForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.inference_mode():
        logits, mtp = model(x)
    print("Hy3 output:", logits.shape, "| MTP output:", None if mtp is None else mtp.shape)
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M (test config)")
