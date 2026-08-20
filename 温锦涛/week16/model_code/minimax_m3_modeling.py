"""
MiniMax-M3 建模文件（教学/结构复现版）
========================================
MiniMax-M3 是 MiniMax 开源的视觉-语言 MoE 模型，约 428B 总参数 / 23B 激活参数，
原生支持 1M 上下文。

本文聚焦其【结构特点】，忠实于官方 config.json 与 MSA 稀疏注意力论文
(arXiv:2606.13392) 的核心设计，忽略训练/并行等工程细节，便于阅读与对比。

三大核心结构创新：
1. MSA (MiniMax Sparse Attention) —— 双分支 blockwise 稀疏注意力
   * Index Branch：以 O(T) 代价为每个 query 评分并挑选 top-16 个 KV block
   * Main Branch ：仅对被选中的 block 执行精确注意力
   * 不改动 Q/K/V/O 投影权重，仅在注意力计算路径插入轻量 Indexer
2. 分层"全注意力 + 稀疏注意力"混合
   * 层 0-2  : Full Attention (GQA 16:1) + Dense FFN
   * 层 3-59 : MSA Sparse Attention + MoE (128 experts, top-4, sigmoid)
3. 7-MTP 多 token 预测投机解码 + 原生视觉编码器 (CLIP ViT-32L + 3D RoPE)

关键超参 (text_config)：
    hidden_size=6144, layers=60, heads=64, kv_heads=4 (GQA 16:1), head_dim=128
    Partial RoPE: theta=5e6, rotary_dim=64 (partial_rotary_factor=0.5)
    MoE: 128 local experts, 4/tok, 1 shared, sigmoid routing, inter=3072
    MSA: index_dim=128, 4 index heads, topk_blocks=16, block_size=128
    vocab=200064, 1M context
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. 基础组件：RMSNorm / 线性层 / RoPE
# ============================================================================

class RMSNorm(nn.Module):
    """Gemma-style RMSNorm（按 use_gemma_norm=True）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(orig_dtype)


class Linear(nn.Module):
    """简单线性层包装，便于统一接口。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _build_rotary_index(dim: int, max_seq_len: int, base: float) -> torch.Tensor:
    """预计算 RoPE 频率（复数形式）。rotary_dim 由外部传入。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, rotary_dim: int):
    """对张量 x 的尾部 rotary_dim 维度应用 RoPE（Partial RoPE）。
    兼容 4D [b, s, h, d] 与 2D [b, s, d] 输入。"""
    x_rope = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    xc = torch.view_as_complex(x_rope.float().reshape(*x_rope.shape[:-1], -1, 2))
    # 依据输入维度调整 freqs 视图
    if x.ndim == 4:
        freqs = freqs_cis.view(1, -1, 1, xc.shape[-1])
    else:  # 2D
        freqs = freqs_cis.view(1, -1, xc.shape[-1])
    out = torch.view_as_real(xc * freqs).flatten(-2)
    return torch.cat([out.to(x.dtype), x_pass], dim=-1)


# ============================================================================
# 2. SwiGLU-OAI 激活函数（hidden_act="swigluoai"）
#    MiniMax 变体：SiLU(gate) 经过 alpha 缩放并 clamp 到 limit
# ============================================================================

class SwiGLUOAI(nn.Module):
    """SwiGLU-OAI：gate = sigmoid(alpha * gate) 风格的分段近似，配合 clamp 稳定训练。"""

    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        # gate 通道经 alpha 缩放后 sigmoid（近似 SiLU），限制上界保证数值稳定
        gate = torch.clamp(gate, max=self.limit)
        # MiniMax 使用的 swigluoai：用 alpha 缩放替代默认系数
        return (torch.sigmoid(self.alpha * gate) * up)


class DenseFFN(nn.Module):
    """Dense FFN（前 3 层使用），SwiGLU-OAI，中间维 12288。"""

    def __init__(self, hidden_size: int, intermediate_size: int, alpha=1.702, limit=7.0):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size)
        self.up_proj = Linear(hidden_size, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_size)
        self.act = SwiGLUOAI(alpha, limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x), self.up_proj(x)))


# ============================================================================
# 3. MSA 稀疏注意力（核心创新）
#    Index Branch (O(T) 选 top-16 block) + Main Branch (只算选中的 block)
# ============================================================================

@dataclass
class SparseAttentionConfig:
    """对应 config 中 sparse_attention_config 字段。"""
    use_sparse_attention: bool = True
    sparse_index_dim: int = 128          # index 分支的 head_dim
    sparse_num_index_heads: int = 4      # index heads 数量
    sparse_topk_blocks: int = 16         # 每 query 选 top-16 个 block
    sparse_block_size: int = 128         # KV 分组 block 大小
    sparse_score_type: str = "max"       # block score 聚合方式
    sparse_init_block: int = 0
    sparse_local_block: int = 1          # 强制包含当前 query 所在 block


class MSAIndexer(nn.Module):
    """
    Index Branch：轻量、低秩的 block 选择器。
    为每个 query 评分并挑选 top-k 个 KV block，供 Main Branch 只做局部注意力。

    结构（不修改主注意力的 Q/K/V/O 权重）：
      - Index Q 投影: hidden -> index_heads * index_dim
      - Index K 投影: hidden -> index_dim（单 key 向量被多个 index heads 共享）
      - QK Norm: per-head Gemma-style RMSNorm
      - Partial RoPE 只作用于 index 向量
    """

    def __init__(self, hidden_size: int, index_heads: int, index_dim: int,
                 block_size: int, topk_blocks: int, local_blocks: int,
                 rope_theta: float, rotary_dim: int, max_seq_len: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.index_heads = index_heads
        self.index_dim = index_dim
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.local_blocks = local_blocks
        # index 向量实际旋转维度不超过其自身维度（Partial RoPE）
        self.rotary_dim = min(rotary_dim, index_dim)

        # Index Q：独立于主注意力 Q 投影（6144 -> index_heads*index_dim = 4*128=512）
        self.index_q_proj = Linear(hidden_size, index_heads * index_dim)
        # Index K：独立于主注意力 K 投影（6144 -> index_dim = 128）
        self.index_k_proj = Linear(hidden_size, index_dim)
        # per-head QK Norm（Gemma-style）
        self.q_norm = RMSNorm(index_dim, eps)
        self.k_norm = RMSNorm(index_dim, eps)
        self.softmax_scale = index_dim ** -0.5

        self.register_buffer("freqs_cis",
                             _build_rotary_index(self.rotary_dim, max_seq_len, rope_theta),
                             persistent=False)

    def _score_blocks(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Blockwise QK 评分。

        q: [b, s, index_heads, index_dim]
        k: [b, t, index_dim]（被所有 index heads 共享）
        输出 block_scores: [b, s, n_blocks]
        """
        b, s, h, d = q.shape
        t = k.shape[1]
        n_blocks = math.ceil(t / self.block_size)
        # 对 key 序列 padding 到 block_size 的整数倍，便于按 block 分组
        pad_len = n_blocks * self.block_size - t
        k_padded = F.pad(k, (0, 0, 0, pad_len)) if pad_len > 0 else k
        k_blocks = k_padded.view(b, n_blocks, self.block_size, d)
        # k 按 block 分组后 amax-pool 聚合（sparse_score_type="max"）
        k_pooled = k_blocks.amax(dim=2)          # [b, n_blocks, d]
        # blockwise QK 打分（float32 精度）
        scores = torch.einsum("bshd,bnd->bshn", q.float(), k_pooled.float())
        # 聚合 heads -> block score（per-head amax 后再按 head 平均）
        block_scores = scores.amax(dim=2)        # [b, s, n_blocks]  "max" over heads
        return block_scores

    def forward(self, hidden: torch.Tensor, index_kv: torch.Tensor,
                start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          topk_block_idx: [b, s, topk_blocks] 选中的 block 下标（含 local block）
          block_scores:   [b, s, n_blocks]    原始 block 分数
        """
        b, s, _ = hidden.shape
        # Step 1: Index Q/K 投影 + QK Norm
        q = self.index_q_proj(hidden).view(b, s, self.index_heads, self.index_dim)
        k = self.index_k_proj(hidden)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Step 2: Partial RoPE（仅 index 前 rotary_dim 维）
        freqs = self.freqs_cis[start_pos:start_pos + s]
        q = _apply_rotary_emb(q, freqs, self.rotary_dim)
        k = _apply_rotary_emb(k, freqs, self.rotary_dim)
        # Step 3: blockwise 评分（含 causal mask）
        block_scores = self._score_blocks(q, k)
        n_blocks = block_scores.shape[-1]
        # causal mask：当前 query 只能看到它所在 block 及之前的 block
        causal_blocks = (torch.arange(s, device=hidden.device).view(-1, 1)
                         + start_pos) // self.block_size
        causal_mask = (torch.arange(n_blocks, device=hidden.device).view(1, -1)
                       <= causal_blocks)                    # [s, n_blocks]
        block_scores = block_scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
        # Step 4: local window boost（强制包含当前 block）
        local_block = torch.arange(s, device=hidden.device) + start_pos
        local_block = (local_block // self.block_size).view(1, -1, 1)
        for _ in range(self.local_blocks):
            block_scores.scatter_(2, local_block, block_scores.max(dim=-1, keepdim=True).values + 1e3)
        # Step 5: top-k block 选择
        topk = min(self.topk_blocks, n_blocks)
        topk_idx = block_scores.topk(topk, dim=-1)[1]       # [b, s, topk]
        return topk_idx, block_scores


class MSAttention(nn.Module):
    """
    MSA 注意力层（Main Branch + Index Branch）。
    对应层 3-59。Q/K/V/O 投影与普通 GQA 完全一致，只多了 index 分支做 block 选择。
    """

    def __init__(self, hidden_size: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 rope_theta: float, rotary_dim: int, max_seq_len: int,
                 index_config: SparseAttentionConfig, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.n_rep = n_heads // n_kv_heads      # GQA 16:1

        # ---- 主注意力投影（与普通 GQA 相同）----
        self.q_proj = Linear(hidden_size, n_heads * head_dim)
        self.k_proj = Linear(hidden_size, n_kv_heads * head_dim)
        self.v_proj = Linear(hidden_size, n_kv_heads * head_dim)
        self.o_proj = Linear(n_heads * head_dim, hidden_size)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        self.softmax_scale = head_dim ** -0.5

        # ---- Index Branch（MSA 核心）----
        self.indexer = MSAIndexer(
            hidden_size,
            index_heads=index_config.sparse_num_index_heads,
            index_dim=index_config.sparse_index_dim,
            block_size=index_config.sparse_block_size,
            topk_blocks=index_config.sparse_topk_blocks,
            local_blocks=index_config.sparse_local_block,
            rope_theta=rope_theta,
            rotary_dim=rotary_dim,
            max_seq_len=max_seq_len,
            eps=eps,
        )
        self.block_size = index_config.sparse_block_size

        self.register_buffer("freqs_cis",
                             _build_rotary_index(rotary_dim, max_seq_len, rope_theta),
                             persistent=False)

    def forward(self, x: torch.Tensor, index_kv: torch.Tensor,
                main_kv: torch.Tensor, start_pos: int, seq_len: int) -> torch.Tensor:
        """
        前向（教学简化，非增量）：
          - x       : 当前输入（已过 input_layernorm）
          - index_kv: 用于 index 分支评分的历史 key（独立缓存）
          - main_kv : 保留接口（真实模型中主 KV 为独立缓存）
        本实现基于 x 直接构造主 KV 与 index 用 KV。
        """
        b, s, _ = x.shape
        freqs = self.freqs_cis[:seq_len]

        # ---- 主注意力 QKV（Q/K/V/O 投影与普通 GQA 完全一致）----
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rotary_emb(q, freqs, self.rotary_dim)
        k = _apply_rotary_emb(k, freqs, self.rotary_dim)

        # ---- Index Branch：O(T) 选 top-16 block（核心创新）----
        topk_block_idx, _ = self.indexer(x, index_kv, start_pos)

        # 构建稀疏注意力掩码：query 只能看选中的 block（含因果 + local block）
        t = index_kv.shape[1]
        n_blocks = math.ceil(t / self.block_size)
        block_mask = torch.zeros(b, s, n_blocks, dtype=torch.bool, device=x.device)
        block_mask.scatter_(2, topk_block_idx, True)
        # 展开到 token 粒度；截断到实际 token 数 t
        sparse_mask = block_mask.repeat_interleave(self.block_size, dim=-1)[:, :, :t]  # [b,s,t]

        # ---- 主注意力：只算被选中的 block，其余 mask 掉 ----
        k = k.repeat_interleave(self.n_rep, dim=2)   # [b, s, n_heads, d]
        v = v.repeat_interleave(self.n_rep, dim=2)
        attn = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale   # [b,s,h,t]
        # 因果 mask（shape: [1, s, 1, t]）
        causal = torch.tril(torch.ones(s, seq_len, dtype=torch.bool, device=x.device))
        attn = attn.masked_fill(~causal.view(1, s, 1, seq_len), float("-inf"))
        # 稀疏 mask（仅保留选中 block；shape: [b, s, 1, t]，广播到 [b,s,h,t]）
        attn = attn.masked_fill(~sparse_mask.unsqueeze(2), float("-inf"))
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.einsum("bsht,bthd->bshd", attn, v)
        out = out.reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(out)


class FullAttention(nn.Module):
    """普通 GQA 注意力（层 0-2 使用，全注意力）。"""

    def __init__(self, hidden_size, n_heads, n_kv_heads, head_dim,
                 rope_theta, rotary_dim, max_seq_len, eps=1e-6):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.n_rep = n_heads // n_kv_heads
        self.q_proj = Linear(hidden_size, n_heads * head_dim)
        self.k_proj = Linear(hidden_size, n_kv_heads * head_dim)
        self.v_proj = Linear(hidden_size, n_kv_heads * head_dim)
        self.o_proj = Linear(n_heads * head_dim, hidden_size)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        self.softmax_scale = head_dim ** -0.5
        self.register_buffer("freqs_cis",
                             _build_rotary_index(rotary_dim, max_seq_len, rope_theta),
                             persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, seq_len: int) -> torch.Tensor:
        b, s, _ = x.shape
        freqs = self.freqs_cis[:seq_len]
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rotary_emb(q, freqs, self.rotary_dim)
        k = _apply_rotary_emb(k, freqs, self.rotary_dim)
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)
        attn = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
        causal = torch.tril(torch.ones(s, seq_len, dtype=torch.bool, device=x.device))
        attn = attn.masked_fill(~causal.view(1, s, 1, seq_len), float("-inf"))
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.einsum("bsht,bthd->bshd", attn, v).reshape(b, s, -1)
        return self.o_proj(out)


# ============================================================================
# 4. MoE（sigmoid 路由，128 experts, top-4, 1 shared）
# ============================================================================

class Expert(nn.Module):
    """单个 MoE 专家 FFN：SwiGLU-OAI，中间维 3072。"""

    def __init__(self, hidden_size: int, inter_size: int, alpha=1.702, limit=7.0):
        super().__init__()
        self.gate_proj = Linear(hidden_size, inter_size)
        self.up_proj = Linear(hidden_size, inter_size)
        self.down_proj = Linear(inter_size, hidden_size)
        self.act = SwiGLUOAI(alpha, limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x), self.up_proj(x)))


class MoE(nn.Module):
    """
    MoE 层：sigmoid 路由（scoring_func="sigmoid"）。
    - 每个 token 选 top-4 routed experts
    - 额外 1 个 shared expert 总是被激活
    - use_routing_bias=True：bias 只影响专家选择，不影响路由权重
    - routed_scaling_factor=2.0
    """

    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int,
                 n_shared_experts: int, inter_size: int, scaling_factor: float,
                 alpha=1.702, limit=7.0):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.scaling_factor = scaling_factor
        self.gate = Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList(
            [Expert(hidden_size, inter_size, alpha, limit) for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList(
            [Expert(hidden_size, inter_size, alpha, limit) for _ in range(n_shared_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        flat = x.reshape(-1, x.shape[-1])
        # sigmoid 路由分数
        scores = self.gate(flat)                        # [b*s, num_experts]
        probs = torch.sigmoid(scores.float())
        topk_probs, topk_idx = probs.topk(self.num_experts_per_tok, dim=-1)
        # L1 归一化路由权重（bias 只影响选择不影响权重）
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        topk_probs = topk_probs * self.scaling_factor

        out = torch.zeros_like(flat, dtype=torch.float32)
        for e in range(self.num_experts):
            mask = (topk_idx == e)
            if mask.any():
                token_ids, slot = torch.where(mask)
                expert_out = self.experts[e](flat[token_ids])
                out[token_ids] += expert_out * topk_probs[token_ids, slot, None]
        # shared experts
        for sh in self.shared_experts:
            out += sh(flat).float()
        return out.reshape(b, s, -1).to(x.dtype)


# ============================================================================
# 5. Transformer Block + 完整模型
# ============================================================================

class Block(nn.Module):
    """统一的 Transformer 层：按 layer_id 选择 Attention 与 FFN 类型。"""

    def __init__(self, layer_id: int, cfg: "MiniMaxM3Config"):
        super().__init__()
        self.layer_id = layer_id
        self.use_moe = cfg.moe_layer_freq[layer_id] == 1
        self.use_sparse = cfg.sparse_attention_freq[layer_id] == 1

        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        if self.use_sparse:
            self.attention = MSAttention(
                cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                cfg.head_dim, cfg.rope_theta, cfg.rotary_dim, cfg.max_position_embeddings,
                cfg.sparse_attention_config, cfg.rms_norm_eps)
        else:
            self.attention = FullAttention(
                cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                cfg.head_dim, cfg.rope_theta, cfg.rotary_dim, cfg.max_position_embeddings,
                cfg.rms_norm_eps)

        if self.use_moe:
            self.mlp = MoE(
                cfg.hidden_size, cfg.num_local_experts, cfg.num_experts_per_tok,
                cfg.n_shared_experts, cfg.shared_intermediate_size,
                cfg.routed_scaling_factor, cfg.swiglu_alpha, cfg.swiglu_limit)
        else:
            self.mlp = DenseFFN(cfg.hidden_size, cfg.dense_intermediate_size,
                                cfg.swiglu_alpha, cfg.swiglu_limit)

    def forward(self, x: torch.Tensor, index_kv: torch.Tensor, main_kv: torch.Tensor,
                start_pos: int, seq_len: int) -> torch.Tensor:
        h = self.input_layernorm(x)
        if self.use_sparse:
            a = self.attention(h, index_kv, main_kv, start_pos, seq_len)
        else:
            a = self.attention(h, start_pos, seq_len)
        h = x + a
        m = self.mlp(self.post_attention_layernorm(h))
        return h + m


@dataclass
class MiniMaxM3Config:
    """MiniMax-M3 文本骨干配置（对应 text_config）。"""
    hidden_size: int = 6144
    intermediate_size: int = 3072
    num_hidden_layers: int = 60
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 200064
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000
    rotary_dim: int = 64
    hidden_act: str = "swigluoai"
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    dense_intermediate_size: int = 12288
    shared_intermediate_size: int = 3072
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    routed_scaling_factor: float = 2.0
    num_mtp_modules: int = 7
    # 每层类型
    moe_layer_freq: List[int] = field(
        default_factory=lambda: [0, 0, 0] + [1] * 57)          # 层0-2 dense, 3-59 moe
    sparse_attention_freq: List[int] = field(
        default_factory=lambda: [0, 0, 0] + [1] * 57)          # 层0-2 full, 3-59 sparse
    sparse_attention_config: SparseAttentionConfig = field(
        default_factory=SparseAttentionConfig)


class MiniMaxM3ForCausalLM(nn.Module):
    """完整 MiniMax-M3 文本骨干（不含视觉编码器，聚焦结构）。"""

    def __init__(self, cfg: MiniMaxM3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(i, cfg) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # 模拟 KV 缓存（教学用：主 KV + index KV 分离）
        max_blocks = cfg.max_position_embeddings // cfg.sparse_attention_config.sparse_block_size + 1
        self.register_buffer("main_kv_k", torch.zeros(1, cfg.max_position_embeddings, cfg.head_dim, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("main_kv_v", torch.zeros(1, cfg.max_position_embeddings, cfg.head_dim, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("index_kv", torch.zeros(1, cfg.max_position_embeddings, cfg.sparse_attention_config.sparse_index_dim, dtype=torch.bfloat16), persistent=False)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        """
        前向演示（非增量、教学简化）。
        每层注意力独立基于自身输入构造 main KV 与 index KV，
        用于直观展示 MSA"双分支"的结构差异，而非模拟真实缓存共享。
        """
        b, s = input_ids.shape
        seq_len = s
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            if layer.use_sparse:
                # 教学简化：用当前输入构造 index key 与 main KV（真实模型用独立缓存）
                hidden_for_kv = layer.input_layernorm(h)
                index_kv = layer.attention.indexer.index_k_proj(hidden_for_kv)   # [b, s, index_dim]
                h = layer(h, index_kv, None, 0, seq_len)
            else:
                h = layer(h, None, None, 0, seq_len)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    # 小规模测试配置（真实模型：hidden=6144, 60层, 128专家, vocab=200064）
    cfg = MiniMaxM3Config(
        hidden_size=128, vocab_size=4096, num_hidden_layers=6,
        dense_intermediate_size=256, shared_intermediate_size=64,
        num_local_experts=8, num_experts_per_tok=2)
    cfg.moe_layer_freq = [0, 0, 0, 1, 1, 1]
    cfg.sparse_attention_freq = [0, 0, 0, 1, 1, 1]
    cfg.sparse_attention_config.sparse_block_size = 4
    cfg.sparse_attention_config.sparse_index_dim = 16
    model = MiniMaxM3ForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.inference_mode():
        logits = model(x)
    print("MiniMax-M3 output:", logits.shape)
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M (test config)")
