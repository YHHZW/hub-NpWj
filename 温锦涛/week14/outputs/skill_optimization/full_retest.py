# -*- coding: utf-8 -*-
"""
完整评估集复测：在 60 题 eval_set 上对比 skill 库三种状态。

三种状态（都基于主 skills/ 的 refund + vip_benefits）：
  1. baseline : 无 digital_goods_refund（现状）
  2. initial  : 加 初始版 digital_goods_refund
  3. optimized: 加 优化版 digital_goods_refund

对比维度：整体/分类准确率 + 平均 input token。

用法：
  需 .env 中已配置 DEEPSEEK_API_KEY
  cd self_evolving_agent
  python outputs/skill_optimization/full_retest.py

结果写入 outputs/skill_optimization/full_retest_result.json
"""
import os
import sys
import json
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import tiktoken

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from openai import OpenAI
from evaluator import Evaluator

# baseline 用 skills_original（refund+vip_benefits 初始版），保证三种状态唯一差异是 digital_goods 版本
BASE_SKILLS = ROOT / "outputs" / "skills_original"
INITIAL_DIR = Path(__file__).parent / "initial" / "digital_goods_refund"
OPTIM_DIR = Path(__file__).parent / "optimized" / "digital_goods_refund"
EVAL_SET = ROOT / "data" / "eval_set.json"
RESULT_FILE = Path(__file__).parent / "full_retest_result.json"

SYSTEM_TEMPLATE = """你是云购商城的智能客服助手。

你的所有知识来源于以下技能文档，严格基于文档内容回答，不要自行推断或编造政策。

## 回答规则（严格遵守）
- 【能回答】如果技能文档覆盖了用户问题：直接给出完整具体的答案（含具体天数/金额/
  工作日数等政策细节）。**不要在答案中加"建议联系人工客服"之类的推脱话**。
- 【不能回答】如果技能文档确实不覆盖：**仅回答一句** "需要联系人工客服"，
  不要编造答案，也不要列举可能的情况。

## 当前知识库（共{count}个技能）

{skills_content}
"""


def load_env_file():
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k and not os.getenv(k):
            os.environ[k] = v


load_env_file()


def read_skills(skill_dirs) -> dict:
    """
    读取 skill 集合。支持两种形态：
      - 集合目录（如 skills/，含多个子目录每个含 SKILL.md）
      - 单个 skill 目录（如 skills/refund，直接含 SKILL.md）
    """
    skills = {}
    for d in skill_dirs:
        if not d.is_dir():
            continue
        direct = d / "SKILL.md"
        if direct.exists():
            skills[d.name] = direct.read_text(encoding="utf-8")
            continue
        for sub in d.iterdir():
            if sub.is_dir():
                f = sub / "SKILL.md"
                if f.exists():
                    skills[sub.name] = f.read_text(encoding="utf-8")
    return skills


def build_prompt(skills: dict) -> str:
    parts = [f"### 技能：{name}\n{content}" for name, content in sorted(skills.items())]
    skills_content = "\n\n---\n\n".join(parts)
    return SYSTEM_TEMPLATE.format(count=len(skills), skills_content=skills_content)


def run_round(label: str, skills: dict, evaluator: Evaluator, client, enc, model: str) -> dict:
    """跑完整 eval_set，返回聚合结果。"""
    prompt = build_prompt(skills)
    enc_prompt = len(enc.encode(prompt, disallowed_special=()))

    total = correct = 0
    sum_prompt = sum_output = 0
    by_category = {}
    errors = []

    for qid, q in sorted(evaluator.questions.items()):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": q["question"]},
            ],
            temperature=0,
            max_tokens=400,
        )
        answer = resp.choices[0].message.content.strip()
        sum_prompt += resp.usage.prompt_tokens
        sum_output += resp.usage.completion_tokens

        ok, reason = evaluator.evaluate_answer(answer, qid)
        total += 1
        cat = q["category"]
        bc = by_category.setdefault(cat, {"total": 0, "correct": 0})
        bc["total"] += 1
        if ok:
            correct += 1
            bc["correct"] += 1
        else:
            errors.append({"id": qid, "category": cat, "reason": reason, "question": q["question"][:50]})

    for c in by_category.values():
        c["accuracy"] = round(c["correct"] / c["total"], 3)

    return {
        "label": label,
        "accuracy": round(correct / total, 3),
        "correct": correct,
        "total": total,
        "avg_prompt_tokens": round(sum_prompt / total, 1),
        "avg_output_tokens": round(sum_output / total, 1),
        "sum_prompt_tokens": sum_prompt,
        "enc_prompt_tokens": enc_prompt,
        "by_category": by_category,
        "errors": errors,
    }


def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误: 未找到 DEEPSEEK_API_KEY，请在项目根目录 .env 中配置后重试")
        sys.exit(1)

    model = os.getenv("LLM_MODEL", "deepseek-chat")
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    enc = tiktoken.get_encoding("cl100k_base")
    evaluator = Evaluator(str(EVAL_SET))

    base = read_skills([BASE_SKILLS])
    initial_dg = read_skills([INITIAL_DIR])
    optim_dg = read_skills([OPTIM_DIR])

    states = {
        "baseline": {**base},
        "initial": {**base, **initial_dg},
        "optimized": {**base, **optim_dg},
    }

    results = {}
    print(f"模型: {model} | 评估集: {len(evaluator.questions)} 题\n")
    for key, skills in states.items():
        print(f"── 运行 [{key}] ...")
        r = run_round(key, skills, evaluator, client, enc, model)
        results[key] = r
        print(f"   {r['correct']}/{r['total']} = {r['accuracy']:.1%}  |  avg_prompt={r['avg_prompt_tokens']}")

    # ── 汇总对比 ─────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("完整评估集复测：三种状态对比")
    print("=" * 66)
    init, opt = results["initial"], results["optimized"]
    print(f"\n整体准确率:")
    for key in ("baseline", "initial", "optimized"):
        r = results[key]
        print(f"  {key:<10} {r['correct']}/{r['total']} = {r['accuracy']:.1%}")

    print(f"\n优化版 vs 初始版（重点对比）:")
    acc_delta = opt["accuracy"] - init["accuracy"]
    tok_save = init["avg_prompt_tokens"] - opt["avg_prompt_tokens"]
    pct = (1 - opt["avg_prompt_tokens"] / init["avg_prompt_tokens"]) * 100 if init["avg_prompt_tokens"] else 0
    print(f"  准确率:   {init['accuracy']:.1%} → {opt['accuracy']:.1%}  (Δ {acc_delta:+.1%})")
    print(f"  平均input token/题: {init['avg_prompt_tokens']} → {opt['avg_prompt_tokens']}  (省 {tok_save:.1f}, {pct:.1f}%)")

    print(f"\n分类准确率:")
    cats = sorted(set(results["baseline"]["by_category"]) | set(results["optimized"]["by_category"]))
    hdr = f"  {'类别':<18}{'基线':>8}{'初始':>8}{'优化':>8}"
    print(hdr)
    for c in cats:
        b = results["baseline"]["by_category"].get(c, {})
        i = results["initial"]["by_category"].get(c, {})
        o = results["optimized"]["by_category"].get(c, {})
        print(f"  {c:<18}{b.get('correct',0):>4}/{b.get('total',0):<3}{i.get('correct',0):>4}/{i.get('total',0):<3}{o.get('correct',0):>4}/{o.get('total',0):<3}")

    # digital_goods 回归检查
    dg_o = results["optimized"]["by_category"].get("digital_goods", {})
    dg_i = results["initial"]["by_category"].get("digital_goods", {})
    print(f"\n  digital_goods: 初始 {dg_i.get('correct',0)}/{dg_i.get('total',0)} | 优化 {dg_o.get('correct',0)}/{dg_o.get('total',0)}")

    RESULT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 结果已保存: {RESULT_FILE}")


if __name__ == "__main__":
    main()
