# GRPO 数学题强化学习项目（作业版）

> 用 **GRPO（Group Relative Policy Optimization）** 在消费级显卡（RTX 4060 Laptop 8GB）上
> 让 **Qwen2-0.5B-Instruct** 学会解决"带括号的四则混合运算 + 求未知数填空"数学题。
> 本项目的**架构仿照**老师的示例项目（`probe → train → probe → compare` 流水线），但
> **题目生成、难度体系、内容均为自研**。

## 一、与示例项目的差异（自研点）

示例项目做的是**纯加减乘**（个位加 → 两位×两位），本项目的难度递进换成了三个自研轴：

| 轴 | 说明 |
|----|------|
| **运算重数** | 单步 → 两步混合 → 带括号优先 → 双层括号/连算 |
| **括号层级** | 无括号 → 一层括号 → 括号嵌套 |
| **未知数复杂度** | 无未知数（正向计算）→ 单步求未知数 → 含括号的多步求未知数 |

**6 级难度（P1~P6）**：

| 级别 | 题型 | 样例 | 训练集? |
|------|------|------|---------|
| P1 | 单步四则（含整除除法） | `36 ÷ 6`、`15 + 24` | —（泛化） |
| P2 | 两步混合（乘除优先） | `12 × 4 + 8` | √ |
| P3 | 括号优先（一层括号） | `(14 + 6) × 3` | √ |
| P4 | 双层括号 / 连乘 | `(20 - 4) ÷ (2 × 2)` | —（泛化） |
| P5 | 求未知数（单步填空） | `? + 15 = 42` | √ |
| P6 | 求未知数（含括号） | `(? - 8) × 3 = 21` | —（泛化） |

**核心卖点**：P5/P6 的"求未知数填空"要求模型**逆向运算**（从结果反推输入），这是示例
完全没有的新题型，也更贴近真实数学题。泛化评估（P1/P4/P6 不进训练集）用于回答
"没训练过的题型是否也提升了"。

## 二、架构总览

```
probe_math.py → train_rl.py → probe_math.py --model → compare_math.py
（基线摸底定难度）（GRPO 训练）  （同评估集复测）     （对比表+曲线）
```

没有构建步骤，5 个独立脚本 + 1 个兼容补丁，数据通过 `outputs/*.json` 衔接。

```
src/
├── math_problems.py     # 自研核心：6 级难度题目生成、三口径解析、系统提示词（被 probe/train 复用）
├── probe_math.py        # 基线摸底：greedy / pass@k / informative group rate
├── train_rl.py          # GRPO 训练：复合奖励 + 难度配比 + 关键配置
├── compare_math.py      # 对比表 + 训练曲线
├── test_general.py      # 灾难性遗忘检查（基座 vs 全量 vs LoRA）
└── trl_compat.py        # 兼容补丁（trl 0.21 × transformers 5.x，必须最先 import）
```

**数据流（跨文件关键纽带）**：`probe_math.py` 是唯一评估入口，通过 `--model` 切换
基座 / checkpoint。评估对每个 prompt 做 greedy×1 加温度 1.0 采样×K，统计
**informative group rate**（组内 `0 < 正确数 < K` 的比例——GRPO 可学习性核心指标）。
`train_rl.py` 依据基线 probe 的 `loose_informative_group_rate` 选题（`LEVEL_MIX`）。

**共享代码（跨文件理解）**：`make_question(level, rng)` 难度生成、`parse_output(text, answer)`
三口径解析（格式/严格/宽松）、`SYSTEM_PROMPT` 都定义在 `math_problems.py`，被
`train_rl.py` 直接 import（复用，不重复实现）。

**GRPO 关键机制**：无价值网络、无奖励模型——组内 K 条样本奖励的均值/标准差直接归一化
出 advantage；`beta=0` 连参考模型都省掉。奖励设计为复合式：`reward_correct`（宽松解析，
正确给 1.0）+ `reward_format`（格式，权重 0.2）。宽松口径保证格式冷启动期也有正确梯度。

## 三、环境准备

```bash
pip install -r requirements.txt
```

| 依赖 | 版本 | 用途 |
|------|------|------|
| torch | 2.6.0+cu126 | 训练框架 |
| transformers | 5.5.3 | 模型加载（5.x 与 trl 0.21 兼容问题由 `trl_compat.py` 处理） |
| trl | 0.21.0 | GRPOTrainer |
| peft | 0.15.0 | 可选，`--lora` 降级方案 |
| accelerate | 1.5.2 | Trainer 后端 |
| datasets | — | 训练集构建 |
| matplotlib | — | 训练曲线绘图 |

**预训练模型**：`D:\badou\八斗课程\pretrain_models\Qwen2-0.5B-Instruct`（已落盘）。
如路径不同，修改 `src/probe_math.py`、`src/train_rl.py`、`src/test_general.py` 顶部的
`MODEL_PATH`。

**硬件要求**：RTX 4060 Laptop（8GB）验证通过，全量微调峰值约 6GB。显存更小的机器用
`--lora` 降级。

## 四、运行步骤

> 所有命令在**项目根目录**运行（`grpo_math_assignment/`）。

### Step 1：基线摸底

```bash
python src/probe_math.py              # 全量：6 难度 × 50 题，K=8 采样
python src/probe_math.py --quick      # 快速验证：每难度 10 题
```

内部流程：程序化生成 6 个难度的题 → 每个 prompt 做 greedy×1 + 温度 1.0 采样×8 →
输出 greedy 正确率、格式遵循率、pass@8、**informative group rate**。
结果保存到 `outputs/baseline_probe.json`。

**预期**：基线的格式遵循率约 0（模型无视 `<answer>` 指令）；P2/P3/P5 的
informative group rate 应落在 0.3~0.8（有梯度可学）；P6 偏低（太难）。

### Step 2：GRPO 训练

```bash
python src/train_rl.py                       # 完整训练：200 步
python src/train_rl.py --max_steps 3 --tag smoke   # 冒烟测试
python src/train_rl.py --lora                # 显存不足时降级 LoRA
python src/train_rl.py --log_completions     # 打印每步真实采样（调试用）
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--max_steps` | 200 | 优化步数；每步 = 4 prompt × 8 采样 |
| `--n_prompts` | 1000 | 训练集大小（P3 50% / P5 25% / P2 25%） |
| `--lr` | 2e-6 | 全量微调学习率（`--lora` 时自动用 2e-4） |
| `--lora` | 关 | LoRA r=16，注意力四层 |
| `--tag` | 空 | 输出目录后缀，区分实验 |

输出：`outputs/grpo_ckpt/`（checkpoint）+ `outputs/train_log.json`（指标序列）。

**健康的训练标志**：前 25 步 `rewards/reward_correct/mean` 从 ~0.5 升到 0.8+；
格式分先收敛、正确分后爬坡——典型的 RL 动态。若奖励恒 0 或补全长乱码，见 §六。

### Step 3：训练后评估

```bash
python src/probe_math.py --model outputs/grpo_ckpt --out outputs/post_train_probe.json --seed 42
python src/probe_math.py --model outputs/grpo_lora_ckpt --out outputs/post_train_probe_lora.json --seed 42
```

**必须保持 `--seed 42` 与基线一致**，保证评估题完全相同，前后可配对比较。
LoRA checkpoint 只含 adapter 权重，脚本检测到 `adapter_config.json` 会自动
先加载基座再挂载 adapter。

### Step 4：对比分析

```bash
python src/compare_math.py
```

输出：对比表（基线 / 全量 / 可选 LoRA 的格式率、greedy 正确率、pass@8）、逐题样例
对照、训练曲线图 `outputs/figures/train_curves.png`。若 LoRA 文件不存在则自动退化
为两方对比。

### Step 5：灾难性遗忘检查（可选）

```bash
python src/test_general.py
```

用 8 类通用场景（知识问答、翻译、逻辑推理等）对比基座 / 全量 / LoRA，检查 GRPO
强化数学题后是否挤占了通用能力。

## 五、作为模块调用

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "outputs/grpo_ckpt"   # 或基座模型路径，对比行为差异
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda")

msgs = [
    {"role": "system", "content": "你是一个数学助手。用户会给你一道算术或求值题，请计算出结果，并把最终答案放在 <answer> 标签中，例如 <answer>42</answer>。不要输出其他内容。"},
    {"role": "user", "content": "计算：(14 + 6) × 3 = ?"},
]
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = tokenizer(text, return_tensors="pt").to("cuda")
out = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
# 基座模型: '60'          （无格式）
# 训练后:   '<answer>60</answer>'
```

## 六、常见问题

**Q1：`from trl import GRPOTrainer` 报 `No module named 'vllm'`？**
trl 0.21 与 transformers 5.x 的已知不兼容。所有脚本第一行 `import trl_compat` 已修复，
确保从项目根目录运行、不要删这个文件。

**Q2：训练中奖励全为 0、补全全是乱码？**
检查是否打开了 gradient checkpointing——transformers 5.x 下它会让 `generate` 输出损坏。
本项目默认关闭。不要为了省显存重新打开，显存不够用 `--lora`。

**Q3：训练一步后模型输出全变成 `!!!!!`？**
权重被 fp16 训废了。确认 `train_rl.py` 里 `model_init_kwargs={"torch_dtype": "bfloat16"}`
存在。根因：本地 Qwen2-0.5B 的 config.json 写的是 fp16，不显式指定会按 fp16 加载，
AdamW 的 `eps=1e-8` 在 fp16 下溢出为 0。

**Q4：CUDA OOM？**
按顺序尝试：`--lora`；减小 `per_device_train_batch_size`（8→4，同时把
`gradient_accumulation_steps` 4→8 保持每步 prompt 数不变）。

**Q5：想换自己的任务/题型？**
改三处并保持一致：`math_problems.py` 的 `make_question`（题目生成）、`train_rl.py` 的
`LEVEL_MIX`（难度配比）和两个 reward 函数（奖励判定）。先跑 probe 确认
informative group rate 在 0.3~0.8 之间再训练。

**Q6：`epoch` 显示 0.8 没跑完一整轮？**
正常。训练集 1000 题，200 步 × 4 题/步 = 800 题，不到一个 epoch。GRPO 是在线采样算法，
题目是否重复不重要（每题每次都会重新采样 8 条）。

## 七、作业说明

本项目是对老师示例的**模仿 + 自研**：
- 架构（probe→train→probe→compare 流水线、复合奖励、informative group rate 选题）完全复刻示例，
  体现对 GRPO 方法论的理解；
- 题目体系（带括号四则 + 求未知数填空）、难度分级（P1~P6）、命名（`make_question` /
  `DIFFICULTY` 等）全部自研，与示例的"纯加减乘"明显区分；
- 完整跑通后可回答：GRPO 能否让 0.5B 模型同时学会"格式 + 计算 + 逆向求未知数"？
  未训练的难度（P1/P4/P6）是否也有提升（泛化）？
