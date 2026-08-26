"""
基线摸底：Qwen2-0.5B-Instruct 在各难度数学题上的表现（greedy / pass@k / 格式遵循率）。

作用与示例 probe_baseline.py 相同，但换用自研的题目生成模块 math_problems：
  - greedy 单样本：测"确定性能力 + 格式遵循"；
  - 温度 1.0 采样 ×K：测 pass@k 和 informative group rate（GRPO 可学习性的核心指标）；
  - 复合奖励前置验证：看当前模型对 <answer> 格式的遵循率。

关键概念（GRPO 的"可学习甜区"）：
  一个 prompt 采样 K 条，组内有对有错才有非零 advantage；
  全对或全错的组不产生梯度 → informative group rate 是选题难度的核心依据。

使用方式：
  python src/probe_math.py             # 全量摸底（50 题/难度，K=8）
  python src/probe_math.py --quick     # 快速验证（10 题/难度，K=8）

输出：
  outputs/baseline_probe.json
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_problems import DIFFICULTY, SYSTEM_PROMPT, make_question, parse_output

ROOT = Path(__file__).parent.parent
MODEL_PATH = Path(r"D:\badou\八斗课程\pretrain_models\Qwen2-0.5B-Instruct")
OUT_PATH = ROOT / "outputs" / "baseline_probe.json"


def build_prompts(tokenizer, problems):
    """把 (表达式, 答案) 列表转成 chat 格式的 prompt 文本。"""
    texts = []
    for expr, _ in problems:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"计算：{expr} = ?"},
        ]
        texts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    return texts


@torch.no_grad()
def generate(model, tokenizer, texts, do_sample, k=1, batch_size=16, max_new_tokens=64):
    """分批生成。do_sample=True 时每条 prompt 返回 k 个样本，外层列表按 prompt 对齐。"""
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=1.0 if do_sample else None,
            top_p=1.0 if do_sample else None,
            num_return_sequences=k if do_sample else 1,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        if do_sample:  # num_return_sequences 把每条 prompt 的 k 个样本连续排列
            all_outputs.extend(
                decoded[j * k : (j + 1) * k] for j in range(len(batch))
            )
        else:
            all_outputs.extend(decoded)
    return all_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="每难度只跑 10 题，快速验证")
    parser.add_argument("--n", type=int, default=50, help="每个难度级别的题目数")
    parser.add_argument("--k", type=int, default=8, help="pass@k 的采样数（与 GRPO group size 一致）")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH),
                        help="模型路径；训练后评估时传 checkpoint 目录")
    parser.add_argument("--out", type=str, default=str(OUT_PATH), help="结果 JSON 输出路径")
    parser.add_argument("--seed", type=int, default=42, help="题目生成随机种子（评估时换种子避免与训练集重叠）")
    args = parser.parse_args()
    n = 10 if args.quick else args.n

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if (Path(args.model) / "adapter_config.json").exists():
        # LoRA checkpoint 只含 adapter 权重：先加载基座再挂载 adapter
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map="cuda"
        )
        model = PeftModel.from_pretrained(base, args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda"
        )
    model.eval()

    rng = random.Random(args.seed)
    report = {}

    for level in DIFFICULTY:
        t0 = time.time()
        problems = [make_question(level, rng) for _ in range(n)]
        texts = build_prompts(tokenizer, problems)

        # greedy 单样本：测"确定性能力 + 格式遵循"
        greedy_outs = generate(model, tokenizer, texts, do_sample=False)
        greedy_fmt = greedy_strict = greedy_loose = 0
        for (expr, ans), out in zip(problems, greedy_outs):
            fmt, strict, loose = parse_output(out, ans)
            greedy_fmt += fmt
            greedy_strict += strict
            greedy_loose += loose

        # 温度采样 k 条：测 pass@k 和 informative group rate（严格/宽松两口径）
        sample_outs = generate(model, tokenizer, texts, do_sample=True, k=args.k)
        sample_strict_sum = sample_loose_sum = 0
        pass_at_k = loose_pass_at_k = 0
        mixed_groups = loose_mixed_groups = 0  # 0 < 正确数 < k：能学到东西的组
        for (_, ans), outs in zip(problems, sample_outs):
            results = [parse_output(o, ans) for o in outs]
            n_strict = sum(r[1] for r in results)
            n_loose = sum(r[2] for r in results)
            sample_strict_sum += n_strict
            sample_loose_sum += n_loose
            pass_at_k += n_strict > 0
            loose_pass_at_k += n_loose > 0
            mixed_groups += 0 < n_strict < args.k
            loose_mixed_groups += 0 < n_loose < args.k

        report[level] = {
            "n": n,
            "k": args.k,
            "greedy_format_rate": round(greedy_fmt / n, 4),
            "greedy_strict_acc": round(greedy_strict / n, 4),
            "greedy_loose_acc": round(greedy_loose / n, 4),
            "sample_strict_acc": round(sample_strict_sum / (n * args.k), 4),
            "sample_loose_acc": round(sample_loose_sum / (n * args.k), 4),
            f"pass@{args.k}": round(pass_at_k / n, 4),
            f"loose_pass@{args.k}": round(loose_pass_at_k / n, 4),
            "informative_group_rate": round(mixed_groups / n, 4),
            "loose_informative_group_rate": round(loose_mixed_groups / n, 4),
            "elapsed_sec": round(time.time() - t0, 1),
            "examples": [
                {"expr": expr, "answer": ans, "greedy_output": out}
                for (expr, ans), out in list(zip(problems, greedy_outs))[:3]
            ],
        }
        r = report[level]
        print(
            f"{level:<22} greedy_loose={r['greedy_loose_acc']:.2f} "
            f"fmt={r['greedy_format_rate']:.2f} "
            f"loose_acc={r['sample_loose_acc']:.2f} "
            f"loose_pass@{args.k}={r[f'loose_pass@{args.k}']:.2f} "
            f"loose_informative={r['loose_informative_group_rate']:.2f} "
            f"({r['elapsed_sec']}s)"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n结果已保存：{out_path}")
    print(f"GPU 峰值显存：{peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
