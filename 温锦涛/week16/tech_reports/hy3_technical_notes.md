# 腾讯混元 Hy3 技术资料（官方信息整理）

> 说明：Hy3 未发布独立 arXiv 技术报告。本文件整理自官方渠道信息，
> 供本项目"大模型结构演进"调研使用。
>
> 来源：
> - 官方仓库 README：GitHub `Tencent-Hunyuan/Hy3`
> - 官方模型卡：HuggingFace `tencent/Hy3`；ModelScope `Tencent-Hunyuan/Hy3`
> - 官方发布页：hy.tencent.com/research/hy3
>
> 整理日期：2026-08-20

## 1. 模型概览

Hy3 是腾讯混元团队的**快慢思考融合混合专家（MoE）**大模型，主打高性价比推理。

- 架构：`HYV3ForCausalLM`（model_type: `hy_v3`）
- 总参数量：295B
- 激活参数量：21B（每次推理仅激活约 7% 参数）
- MTP 层参数：3.8B
- 上下文长度：256K（262,144 tokens）
- 开源协议：Apache 2.0

## 2. 架构配置（来自官方 config.json）

| 参数 | 值 |
|---|---|
| hidden_size | 4096 |
| num_hidden_layers | 80 |
| num_attention_heads | 64 |
| num_key_value_heads | 8（GQA 8:1） |
| head_dim | 128 |
| vocab_size | 120832 |
| max_position_embeddings | 262144 |
| rope_theta | 11,158,840 |
| qk_norm | true |
| rms_norm_eps | 1e-5 |
| intermediate_size（dense FFN） | 13312 |
| first_k_dense_replace | 1（前 1 层 dense） |
| num_experts | 192 |
| num_experts_per_tok | 8 |
| num_shared_experts | 1 |
| moe_intermediate_size（expert_hidden_dim） | 1536 |
| router_scaling_factor | 2.826 |
| moe_router_use_sigmoid | true |
| moe_router_enable_expert_bias | true |
| num_nextn_predict_layers | 1（MTP） |
| 精度 | BF16（另有 Hy3-FP8 量化版） |

## 3. 关键结构特点

### 3.1 大规模稀疏 MoE
- 192 路由专家 + 1 共享专家，每 token 激活 top-8
- sigmoid 路由 + expert bias（bias 影响选择不影响权重）
- route_norm + router_scaling_factor=2.826

### 3.2 快慢思考融合（Hybrid Thinking）
通过 `reasoning_effort` 参数路由推理深度：
- `no_think`：直接回复
- `low`：低深度思考
- `high`：深度链式思考（适合数学/编程/推理）

### 3.3 MTP 多 token 预测
1 层 MTP（3.8B 参数），用于投机解码，提升推理吞吐。

## 4. 性能表现（官方 README）

- 在 SWE-Bench Verified 等编码/推理基准上优于同类规模模型
- 与 GLM-5.1 盲评对比：Hy3 2.67/4 vs GLM-5.1 2.51/4
- 幻觉率 12.5% → 5.4%（内部评测）
- 多轮对话问题率 17.4% → 7.9%

## 5. 开源资源

- GitHub：https://github.com/Tencent-Hunyuan/Hy3
- HuggingFace：https://huggingface.co/tencent/Hy3
- ModelScope：https://modelscope.cn/models/Tencent-Hunyuan/Hy3
- 部署：vLLM recipes / SGLang cookbook
- 量化工具：https://github.com/tencent/AngelSlim
