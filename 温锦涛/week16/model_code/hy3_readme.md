# 腾讯混元 Hy3 结构特点调研

> 调研用途：大模型结构演进课程（周 16）
> 技术资料：Hy3 无独立 arXiv 论文，官方技术说明见 GitHub `Tencent-Hunyuan/Hy3` README，
> 已整理存至 `tech_reports/hy3_technical_notes.md`
> 源码：`hy3_modeling.py` + `hy3_config.json`

## 1. 模型定位

Hy3 是腾讯混元团队开源的**快慢思考融合 MoE** 大模型，总参 **295B / 激活 21B**，
MTP 层 3.8B，Apache 2.0 协议开源，支持 256K 上下文。

| 维度 | 数值 |
|---|---|
| 架构 | HYV3ForCausalLM（hy_v3） |
| 总参数 / 激活 | 295B / 21B（激活仅约 7%） |
| 层数 | 80 + 1 MTP |
| 上下文 | 262,144 (256K) |
| 开源协议 | Apache 2.0 |

## 2. 核心结构特点

### 2.1 大规模稀疏 MoE（高稀疏比）

Hy3 的核心在于**极高的 MoE 稀疏比**：192 个路由专家，每 token 只激活 top-8，
配合 1 个共享专家。激活参数仅占总参的 7% 左右，实现"大容量 + 低算力"。

```
  token ──▶ Gate(sigmoid) ──▶ top-8 路由专家 ──┐
             │                  + 1 共享专家     ├──▶ 输出
             └──── 192 专家池（只激活 8/192）────┘
```

- **sigmoid 路由**（`moe_router_use_sigmoid=True`）：每个专家独立打分，不互相约束
- **expert bias**（`moe_router_enable_expert_bias=True`）：bias 只影响专家选择，不影响权重
- **route_norm**：top-8 权重 L1 归一化
- **router_scaling_factor=2.826**

### 2.2 快慢思考融合（Hybrid Thinking）

Hy3 通过 `reasoning_effort`（`no_think` / `low` / `high`）路由推理深度：

| 模式 | 推理深度 | 适用场景 |
|---|---|---|
| no_think（默认） | 1-2 步直接推理 | 简单问答、信息抽取、翻译 |
| low | 浅层思考 | 一般任务 |
| high | 深度链式思考 | 数学、编程、复杂 Agent |

> 说明：快慢思考融合本质是**推理范式/训练目标**的设计，通过特殊 token 触发，
> 不改变 Transformer 骨干结构。这与 Qwen3 的 think/nothink、DeepSeek 的
> think-high/max 是同一类"推理深度路由"工程实践。

### 2.3 MTP 多 token 预测

1 层 MTP（`num_nextn_predict_layers=1`，3.8B 参数），投机解码提升推理吞吐，
与 DeepSeek-V4、Qwen3.6、GLM-5.2 的 MTP 思路一致。

## 3. 关键超参（config.json）

| 参数 | 值 | 说明 |
|---|---|---|
| hidden_size | 4096 | |
| num_hidden_layers | 80 | |
| num_attention_heads / kv | 64 / 8 | GQA 8:1 |
| head_dim | 128 | |
| vocab_size | 120832 | |
| max_position_embeddings | 262144 | 256K |
| rope_theta | 11,158,840 | |
| qk_norm | true | |
| rms_norm_eps | 1e-5 | |
| first_k_dense_replace | 1 | 前 1 层 dense |
| intermediate_size | 13312 | dense FFN 中间维 |
| num_experts | 192 | 路由专家 |
| num_experts_per_tok | 8 | top-8 |
| num_shared_experts | 1 | |
| moe_intermediate_size | 1536 | 专家中间维（expert_hidden_dim） |
| router_scaling_factor | 2.826 | |
| num_nextn_predict_layers | 1 | MTP |

## 4. 与本项目已调研模型的对比

| 维度 | 腾讯 Hy3 | MiniMax-M3 | DeepSeek-V4 | Kimi-K3 | Qwen3.6 |
|---|---|---|---|---|---|
| 总参/激活 | 295B/21B | 428B/23B | 1.6T/49B | 2.8T/104B | 35B/3B |
| 激活占比 | ~7% | ~5.4% | ~3% | ~3.7% | ~8.6% |
| MoE 专家/激活 | 192/top-8 | 128/top-4 | 256/top-6 | 896/16 | 256/8+1 |
| 路由函数 | sigmoid | sigmoid | sqrtsoftplus | — | — |
| 上下文 | 256K | 1M | 1M | 1M | 262K |
| 注意力 | GQA 8:1 | MSA 稀疏 | CSA+HCA | KDA | DeltaNet+Attn |
| 前 N 层 dense | 1 层 | 3 层 | 部分 | 1 层 | — |
| 多模态 | 文本 | V/L | 文本 | V | V |
| MTP | 1 层 | 7-MTP | MTP | — | MTP |
| 推理范式 | 快慢思考融合 | think/non-think | think-high/max | reasoning_effort | think/nothink |

**Hy3 的结构启示**：它代表了"极致稀疏 MoE（激活 7%）+ 快慢思考融合（推理深度路由）"
这条工程化路线——不追求 1M 超长上下文的注意力创新，而是用高稀疏比 MoE 在 256K
上下文内实现低算力、普惠的模型部署。

## 5. 资料与备注

- 官方仓库：GitHub `Tencent-Hunyuan/Hy3`（README 即官方技术说明）
- HuggingFace：huggingface.co/tencent/Hy3；ModelScope：Tencent-Hunyuan/Hy3
- 量化工具：AngelSlim（FP8 版本 Hy3-FP8）
- 注意：Hy3 **无独立 arXiv 技术报告**，本调研基于官方 README 与 config.json 整理，
  并经 ModelScope 元数据交叉验证。
