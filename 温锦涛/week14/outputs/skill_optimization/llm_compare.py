# -*- coding: utf-8 -*-
"""
真实 LLM 对比：digital_goods_refund skill 初始版 vs 优化版。

对比维度：
  1. 准确率（用项目原装 Evaluator 规则评估，digital_goods 类目 12 题）
  2. 真实 input token（DeepSeek 返回的 usage.prompt_tokens）

用法：
  先设置 DEEPSEEK_API_KEY 环境变量
  cd self_evolving_agent
  python outputs/skill_optimization/llm_compare.py

输出：console + outputs/skill_optimization/llm_compare_result.json
"""
import os
import sys
import json
from pathlib import Path

# 保证 Windows 控制台下中文正常输出（避免 GBK 编码崩溃）
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import tiktoken

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from openai import OpenAI


def load_env_file():
    """从项目根目录 .env 读取环境变量（若环境变量未设置）。"""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and not os.getenv(key):
            os.environ[key] = value


load_env_file()

BASE_SKILLS = ROOT / "outputs" / "skills_original"
INITIAL_DIR = Path(__file__).parent / "initial"
OPTIM_DIR = Path(__file__).parent / "optimized"
EVAL_SET = ROOT / "data" / "eval_set.json"
DIGITAL_IDS = list(range(35, 46))  # 35~45 共 12 题
RESULT_FILE = Path(__file__).parent / "llm_compare_result.json"

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


def load_eval():
    data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in data["questions"]}
    return questions


def evaluate_answer(answer: str, question_id: int, questions: dict) -> tuple:
    """复用 evaluator 的契约式规则评估（与 src/evaluator.py 一致）。"""
    import re
    q = questions[question_id]
    gt = q["ground_truth"]
    normalized = re.sub(r"(?<=\d)[,，](?=\d)", "", answer)
    # 推脱一票否决
    if "联系人工" in normalized:
        return False, "Agent 推脱"
    for kw in gt.get("required", []):
        if kw not in normalized:
            return False, f"缺少关键词: {kw}"
    for kw in gt.get("forbidden", []):
        # 否定前置检测
        idx = normalized.find(kw)
        if idx >= 0:
            window = normalized[max(0, idx - 4):idx]
            if any(neg in window for neg in ["不", "无", "非", "未", "没"]):
                continue
            return False, f"出现禁止词: {kw}"
    return True, "correct"


def run_round(label: str, skills: dict, questions: dict, client, enc, model: str) -> dict:
    prompt = build_prompt(skills)
    enc_prompt_tokens = len(enc.encode(prompt, disallowed_special=()))
    total_prompt_tokens = 0
    total_output_tokens = 0
    correct, total = 0, 0
    details = []

    for qid in DIGITAL_IDS:
        q = questions[qid]
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
        usage = resp.usage
        total_prompt_tokens += usage.prompt_tokens
        total_output_tokens += usage.completion_tokens
        ok, reason = evaluate_answer(answer, qid, questions)
        correct += int(ok)
        total += 1
        details.append({
            "id": qid,
            "question": q["question"],
            "answer": answer,
            "correct": ok,
            "reason": reason,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        })

    result = {
        "label": label,
        "accuracy": round(correct / total, 3),
        "correct": correct,
        "total": total,
        "avg_prompt_tokens": round(total_prompt_tokens / total, 1),
        "avg_completion_tokens": round(total_output_tokens / total, 1),
        "sum_prompt_tokens": total_prompt_tokens,
        "sum_completion_tokens": total_output_tokens,
        "enc_prompt_tokens": enc_prompt_tokens,
        "details": details,
    }
    return result


def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误: 未找到 DEEPSEEK_API_KEY")
        print("请在项目根目录 .env 文件中填入你的 DeepSeek API Key：")
        print("  1. 打开 d:/Class/self_evolving_agent/.env")
        print("  2. 将 DEEPSEEK_API_KEY= 后面填入你的 key")
        print("  3. 重新运行本脚本")
        sys.exit(1)

    model = os.getenv("LLM_MODEL", "deepseek-chat")
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    enc = tiktoken.get_encoding("cl100k_base")
    questions = load_eval()

    base = read_skills([BASE_SKILLS])
    initial_dg = read_skills([INITIAL_DIR / "digital_goods_refund"])
    optim_dg = read_skills([OPTIM_DIR / "digital_goods_refund"])

    print(f"模型: {model}")
    print(f"评估题: digital_goods 类目 12 题 (id 35~45)\n")

    r1 = run_round("初始版(verbose)", {**base, **initial_dg}, questions, client, enc, model)
    r2 = run_round("优化版(compact)", {**base, **optim_dg}, questions, client, enc, model)

    # ── 输出 ─────────────────────────────────────────────
    print("=" * 68)
    print("真实 LLM 对比结果")
    print("=" * 68)
    for r in (r1, r2):
        print(f"\n[{r['label']}]")
        print(f"  准确率:   {r['correct']}/{r['total']} = {r['accuracy']:.1%}")
        print(f"  平均 input token/题:  {r['avg_prompt_tokens']}")
        print(f"  平均 output token/题: {r['avg_completion_tokens']}")
        print(f"  system prompt token (tiktoken估算): {r['enc_prompt_tokens']}")

    print("\n" + "-" * 68)
    print("优化前后差异")
    print("-" * 68)
    acc_delta = r2["accuracy"] - r1["accuracy"]
    tok_delta = r1["avg_prompt_tokens"] - r2["avg_prompt_tokens"]
    pct = (1 - r2["avg_prompt_tokens"] / r1["avg_prompt_tokens"]) * 100 if r1["avg_prompt_tokens"] else 0
    print(f"  准确率:   {r1['accuracy']:.1%} → {r2['accuracy']:.1%}   (Δ {acc_delta:+.1%})")
    print(f"  平均 input token/题: {r1['avg_prompt_tokens']} → {r2['avg_prompt_tokens']}  (节省 {tok_delta:.1f} tokens, {pct:.1f}%)")

    RESULT_FILE.write_text(
        json.dumps({"initial": r1, "optimized": r2}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n✓ 结果已保存: {RESULT_FILE}")


if __name__ == "__main__":
    main()
