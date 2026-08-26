"""
题目生成与解析核心模块：6 级难度（带括号四则混合 + 求未知数填空）。

设计思路（区别于纯加减乘的示例）：
  示例的难度递进是"数位 + 运算符"两个轴（个位加 → 两位×两位）。
  本项目的难度递进换成"运算重数 + 括号层级 + 未知数复杂度"三个轴：
    P1  单步四则          （含除法、整除）
    P2  两步混合           （乘除优先，无括号）
    P3  括号优先           （一层括号，先算括号）
    P4  双层括号/连除      （括号嵌套或同优先级连算）
    P5  求未知数（单步）    （? + 15 = 42 这类填空，需逆向运算）
    P6  求未知数（含括号）  （(? - 8) × 3 = 21，需多步逆向）

难点拆解：
  P1/P2/P3/P4 是"正向计算"，模型要练的是运算顺序与进位/借位；
  P5/P6 是"逆向求未知数"，模型要练的是"从结果反推输入"，这是示例完全没有的新题型，
  也更贴近真实数学题。这也让项目有了自己的"卖点"。

解析口径（与训练奖励解耦，三个层次）：
  format  —— 输出是否含 <answer>整数</answer> 标签
  strict  —— 有标签且标签内数字==标准答案
  loose   —— 输出最后一个数字==标准答案（不管格式，宽松，用于冷启动梯度）
"""

import random
import re

# 全局系统提示词：要求模型只输出带 <answer> 标签的结果，不输出过程
SYSTEM_PROMPT = (
    "你是一个数学助手。用户会给你一道算术或求值题，请计算出结果，"
    "并把最终答案放在 <answer> 标签中，例如 <answer>42</answer>。"
    "不要输出其他内容。"
)

# 正则：匹配 <answer>标签内的整数（可负）；及任意整数（宽松口径取最后一个）
TAG_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
NUM_RE = re.compile(r"-?\d+")

# 6 级难度名称（P 开头，区别于示例的 L）
DIFFICULTY = [
    "P1_single_step",      # 单步四则
    "P2_two_step",         # 两步混合
    "P3_bracket",          # 括号优先
    "P4_double_bracket",   # 双层括号/连算
    "P5_unknown_step",     # 求未知数（单步）
    "P6_unknown_bracket",  # 求未知数（含括号）
]


def _make_division(rng: random.Random):
    """生成整除除法：a = d × q，确保 a ÷ d 一定整除。返回 (被除数, 除数, 商)。"""
    d = rng.randint(2, 9)
    q = rng.randint(2, 9)
    return d * q, d, q


def make_question(level: str, rng: random.Random):
    """按难度级别生成一道题，返回 (表达式文本, 标准答案)。"""
    if level == "P1_single_step":
        kind = rng.choice(["+", "-", "×", "÷"])
        if kind == "÷":
            a, d, q = _make_division(rng)
            return f"{a} ÷ {d}", q
        if kind == "+":
            a, b = rng.randint(12, 99), rng.randint(12, 99)
            return f"{a} + {b}", a + b
        if kind == "-":
            a, b = rng.randint(12, 99), rng.randint(12, 99)
            a, b = max(a, b), min(a, b)
            if a == b:  # 避免答案 0，重采样一个
                a += 1
            return f"{a} - {b}", a - b
        a, b = rng.randint(12, 99), rng.randint(3, 9)
        return f"{a} × {b}", a * b

    if level == "P2_two_step":
        # a × b + c  或  a × b - c（乘除优先，无括号）
        a, b = rng.randint(11, 99), rng.randint(2, 9)
        prod = a * b
        c = rng.randint(10, prod - 10)  # 保证减法结果为正
        if rng.random() < 0.5:
            return f"{a} × {b} + {c}", prod + c
        return f"{a} × {b} - {c}", prod - c

    if level == "P3_bracket":
        # (a + b) × c  或  (a - b) × c，一层括号优先
        if rng.random() < 0.5:
            a, b = rng.randint(12, 99), rng.randint(12, 99)
            inner = a + b
            op = "+"
        else:
            # 减法分支强制 a > b，保证括号内结果为正
            a, b = rng.randint(12, 99), rng.randint(12, 99)
            a, b = max(a, b), min(a, b)
            if a == b:  # 避免答案 0，重采样一个
                a += 1
            inner = a - b
            op = "-"
        c = rng.randint(2, 9)
        return f"({a} {op} {b}) × {c}", inner * c

    if level == "P4_double_bracket":
        # 两种：带括号的除法/乘法；或连乘
        if rng.random() < 0.5:
            a, b = rng.randint(40, 99), rng.randint(10, 39)
            a, b = max(a, b), min(a, b)
            inner = a - b
            c = rng.randint(2, 9)
            if inner % c == 0:
                return f"({a} - {b}) ÷ {c}", inner // c
            return f"({a} - {b}) × {c}", inner * c
        a, b, c = rng.randint(3, 9), rng.randint(3, 9), rng.randint(2, 5)
        return f"{a} × {b} × {c}", a * b * c

    if level == "P5_unknown_step":
        # ? + b = c  →  ? = c - b
        # a + ? = c  →  ? = c - a
        # ? - b = c  →  ? = c + b
        # a - ? = c  →  ? = a - c
        # ? × b = c  →  ? = c ÷ b
        # a ÷ ? = c  →  ? = a ÷ c
        kind = rng.choice(["x+b", "a+x", "x-b", "a-x", "x×b", "a÷x"])
        if kind == "x+b":
            b, ans = rng.randint(11, 99), rng.randint(11, 99)
            return f"? + {b} = {ans + b}", ans
        if kind == "a+x":
            a, ans = rng.randint(11, 99), rng.randint(11, 99)
            return f"{a} + ? = {a + ans}", ans
        if kind == "x-b":
            b, ans = rng.randint(1, 50), rng.randint(1, 50)
            return f"? - {b} = {ans}", b + ans
        if kind == "a-x":
            a = rng.randint(30, 99)
            ans = rng.randint(10, a - 10)
            return f"{a} - ? = {a - ans}", ans
        if kind == "x×b":
            b, ans = rng.randint(3, 9), rng.randint(2, 12)
            return f"? × {b} = {b * ans}", ans
        # a ÷ ? = q：先定 ans(?) 和商 q，再构造 a = ans × q，保证整除
        ans = rng.randint(3, 20)
        q = rng.randint(2, 9)
        a = ans * q
        return f"{a} ÷ ? = {q}", ans

    if level == "P6_unknown_bracket":
        # (? - b) × c = d  →  ? = d ÷ c + b
        # (? + b) × c = d  →  ? = d ÷ c - b
        b, c = rng.randint(5, 40), rng.randint(2, 9)
        if rng.random() < 0.5:
            # (? - b) × c = d → ? = d ÷ c + b（始终为正）
            inner = rng.randint(20, 80)
            ans = inner + b
            return f"(? - {b}) × {c} = {inner * c}", ans
        # (? + b) × c = d → ? = d ÷ c - b；强制 inner > b，保证答案为正
        inner = rng.randint(b + 1, 80)
        ans = inner - b
        return f"(? + {b}) × {c} = {inner * c}", ans

    raise ValueError(level)


def parse_output(text: str, answer: int):
    """解析模型输出，返回 (是否符合格式, 严格正确, 宽松正确)。"""
    m = TAG_RE.search(text)
    fmt_ok = m is not None
    strict_ok = fmt_ok and int(m.group(1)) == answer
    nums = NUM_RE.findall(text)
    loose_ok = bool(nums) and int(nums[-1]) == answer  # 宽松：输出最后一个数字正确
    return fmt_ok, strict_ok, loose_ok
