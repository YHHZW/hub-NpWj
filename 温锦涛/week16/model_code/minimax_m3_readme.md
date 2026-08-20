# MiniMax-M3 结构特点调研

> 调研用途：大模型结构演进课程（周 16）
> 对应论文：**MiniMax Sparse Attention (MSA)**，arXiv:2606.13392
> 论文已存至 `tech_reports/minimax_m3_msa_arxiv.pdf`
> 源码：`minimax_m3_modeling.py` + `minimax-m3_config.json`

## 1. 模型定位

MiniMax-M3 是 MiniMax 开源的**视觉-语言 MoE** 模型，约 **428B 总参数 / 23B 激活参数**，
原生支持 **1M 上下文**。它是把"超长上下文稀疏注意力"做到工程落地的一个代表性模型。

| 维度 | 数值 |
|---|---|
| 架构 | Decoder-only Transformer + Vision Encoder |
| 总参数 / 激活 | ~428B / ~23B |
| 层数 | 60 |
| 上下文 | 1,048,576 (1M) |
| 开源协议 | minimax-community |

## 2. 三大核心结构创新

### 2.1 MSA 稀疏注意力（核心）

**MSA (MiniMax Sparse Attention)** 是双分支 **blockwise** 稀疏注意力：

```
                    ┌─────────────────────────────────────────────┐
  hidden ──────────▶│                MSA 注意力层                    │
                    │                                             │
                    │   Q/K/V/O 投影 与普通 GQA 完全一致            │
                    │         │                                  │
                    │         ├──────────────┐                   │
                    │         ▼              ▼                   │
                    │   [Main Branch]   [Index Branch]           │
                    │   精确注意力        O(T) 选 top-16 block    │
                    │   只算选中的 block                          │
                    └─────────────────────────────────────────────┘
```

- **Index Branch**：以 O(T) 代价为每个 query 评分，挑选 **top-16 个 KV block**（每 block 128 token）
- **Main Branch**：仅对被选中的 16×128=**2048 个 token** 执行精确注意力
- **不改动 Transformer 骨干**（Q/K/V/O 投影权重不变），只在注意力路径插入轻量 Indexer

**Indexer 结构**（每层仅约 3.93M 参数，57 层合计 224M，占总参 0.05%）：
- 4 个 index heads，head_dim 128
- Index Q 投影：6144→512；Index K 投影：6144→128（单 key 被 4 头共享）
- QK Norm（Gemma-style）+ Partial RoPE
- 评分聚合方式 `sparse_score_type="max"`（per-block amax pooling）
- `local_blocks=1`：强制包含 query 所在 block

**复杂度收益**（对比 Full Attention @1M 上下文）：
- Prefill：~15.5× 加速（16.38 → 1.058 PFLOPs/层）
- Decode：~30× 加速
- 单 token 计算量降至上一代 1/20

### 2.2 分层混合：全注意力 + 稀疏注意力 + Dense/MoE

| 层范围 | Attention 类型 | FFN 类型 |
|---|---|---|
| 层 0-2 | Full Attention（GQA 16:1） | Dense FFN（SwiGLU-OAI，inter 12288） |
| 层 3-59 | **MSA Sparse Attention** | **MoE**（128 experts, top-4, sigmoid） |

**设计意图**：前 3 层保留 Full Attention——浅层 token 表示尚未分化，稀疏索引可能不稳定，
先用全注意力锚定语义表示质量。

### 2.3 原生视觉编码器 + 7-MTP

- **视觉编码器**：CLIP ViT-32L（hidden 1280，32 层，patch 14，image 2016）+ 3D RoPE（T/H/W 三轴）
- **7-MTP**：7 模块多 token 预测投机解码，提升解码吞吐

## 3. 关键超参（文本骨干）

| 参数 | 值 | 说明 |
|---|---|---|
| hidden_size | 6144 | |
| num_hidden_layers | 60 | |
| num_attention_heads / kv | 64 / 4 | GQA 16:1 |
| head_dim | 128 | |
| vocab_size | 200064 | |
| max_position_embeddings | 1048576 | 1M |
| rope_theta | 5,000,000 | Partial RoPE，rotary_dim=64（partial 0.5） |
| num_local_experts | 128 | 路由专家 |
| num_experts_per_tok | 4 | top-4 |
| n_shared_experts | 1 | 共享专家 |
| scoring_func | sigmoid | 非 softmax |
| shared_intermediate_size | 3072 | 专家中间维 |
| routed_scaling_factor | 2.0 | |
| num_mtp_modules | 7 | MTP |
| swiglu_alpha / limit | 1.702 / 7.0 | SwiGLU-OAI |

## 4. 与本项目已调研模型的对比

| 维度 | MiniMax-M3 | DeepSeek-V4 | Kimi-K3 | Qwen3.6 | GLM-5.2 |
|---|---|---|---|---|---|
| 核心注意力创新 | **MSA 双分支稀疏** | CSA+HCA | KDA+AttnRes | Gated DeltaNet+Attn | IndexShare |
| 稀疏方式 | block 选择（top-16 block） | KV 压缩（压缩率 4~128） | 线性注意+门控 | 线性注意+稀疏 | index 复用 |
| 上下文 | 1M | 1M | 1M | 262K | 1M |
| MoE 专家/激活 | 128/top-4 | 256/top-6 | 896/16 | 256/8+1 | — |
| 路由函数 | sigmoid | sqrtsoftplus | — | — | — |
| 位置编码 | Partial RoPE(theta 5M) | YaRN | — | — | — |
| 多模态 | 原生 V/L | 文本 | 原生 V | 原生 V | 文本 |
| 投机解码 | 7-MTP | MTP | — | MTP | MTP 增强 |

**MiniMax-M3 的结构启示**：它代表了一条与 DeepSeek-V4 的 KV 压缩、Kimi 的线性注意
**不同的技术路线**——用"轻量索引 + block 稀疏注意力"直接把长上下文计算复杂度从
O(T²) 降到 O(T·k·block)，不改骨干、不改 KV 压缩，工程实现最干净。

## 5. 资料与备注

- 官方博客：minimax.io/blog/minimax-m3
- GitHub：MiniMax-AI/MiniMax-M3；MSA 仓库：MiniMax-AI/MSA
- 注意：MiniMax **尚未发布 M3 的独立完整技术报告**，本调研基于 MSA 论文（2606.13392）
  与官方 config.json 整理；训练数据/硬件/并行策略等未公开。
