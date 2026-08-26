"""
训练前后对比：读取两次 probe 结果 + 训练日志，生成对比表、样例对照和训练曲线。

与示例 compare_results.py 的差异：
  - 难度名称用自研 DIFFICULTY（P1~P6）；
  - 训练集难度改为 P2/P3/P5，泛化集为 P1/P4/P6；
  - 曲线标签与字段沿用 trl 日志格式。

使用方式：
  python src/compare_math.py

输入：
  outputs/baseline_probe.json       # 基线（seed=42）
  outputs/post_train_probe.json     # 训练后（同样 seed=42）
  outputs/train_log.json            # GRPO 训练日志

输出：
  outputs/figures/train_curves.png  # 训练曲线
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_problems import DIFFICULTY

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"

TRAINED_LEVELS = {"P3_bracket", "P5_unknown_step", "P2_two_step"}


def fmt_table(reports):
    """reports: [(标签, 该次 probe 的 report dict), ...]，第一个是基线。"""
    header = f"{'难度':<22}{'训练集':^6}"
    for name, _ in reports:
        header += f"{name + ' 格式/正确/pass@8':^30}"
    rows = []
    for lv in DIFFICULTY:
        row = f"{lv:<22}{'√' if lv in TRAINED_LEVELS else '—':^6}"
        for name, rep in reports:
            r = rep[lv]
            row += (
                f"{r['greedy_format_rate']:.2f} / {r['greedy_loose_acc']:.2f} / {r['loose_pass@8']:.2f}"
                .center(30)
            )
        rows.append(row)
    return header + "\n" + "\n".join(rows)


def fmt_examples(base, post, n=3):
    """训练集内难度各取 n 条 greedy 输出对照。"""
    lines = []
    for lv in ["P2_two_step", "P3_bracket", "P5_unknown_step"]:
        lines.append(f"\n--- {lv} ---")
        for eb, ep in zip(base[lv]["examples"][:n], post[lv]["examples"][:n]):
            lines.append(f"  {eb['expr']} = {eb['answer']}")
            lines.append(f"    前: {eb['greedy_output']!r}")
            lines.append(f"    后: {ep['greedy_output']!r}")
    return "\n".join(lines)


def plot_curves(log_entries, fig_path):
    """log_entries: [(标签, log_history), ...]，多条曲线叠加对比。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, log_history in log_entries:
        logs = [e for e in log_history if "reward" in e]
        steps = [e["step"] for e in logs]
        axes[0].plot(steps, [e["rewards/reward_correct/mean"] for e in logs], label=f"{name} correct")
        axes[0].plot(steps, [e["rewards/reward_format/mean"] for e in logs],
                     linestyle="--", label=f"{name} format")
        axes[1].plot(steps, [e["frac_reward_zero_std"] for e in logs], label=name)
        axes[2].plot(steps, [e["entropy"] for e in logs], label=name)

    axes[0].set_title("Reward components (group mean)")
    axes[0].set_xlabel("step")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("frac_reward_zero_std\n(degenerate group ratio)")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].set_title("Policy entropy")
    axes[2].set_xlabel("step")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig_path.parent.mkdir(exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"训练曲线已保存：{fig_path}")


def main():
    with open(OUT / "baseline_probe.json", encoding="utf-8") as f:
        base = json.load(f)
    with open(OUT / "post_train_probe.json", encoding="utf-8") as f:
        post_full = json.load(f)
    with open(OUT / "train_log.json", encoding="utf-8") as f:
        log_full = json.load(f)

    reports = [("基线", base), ("全量", post_full)]
    log_entries = [("full", log_full)]

    # LoRA 实验存在时纳入三方对比
    lora_probe_path = OUT / "post_train_probe_lora.json"
    lora_log_path = OUT / "train_log_lora.json"
    if lora_probe_path.exists():
        with open(lora_probe_path, encoding="utf-8") as f:
            reports.append(("LoRA", json.load(f)))
    if lora_log_path.exists():
        with open(lora_log_path, encoding="utf-8") as f:
            log_entries.append(("lora", json.load(f)))

    print("=" * 96)
    print("训练前后对比（同一评估集，seed=42，50 题/难度；格式率 / greedy正确率 / pass@8）")
    print("=" * 96)
    print(fmt_table(reports))
    print("\n" + "=" * 96)
    print("样例对照（greedy 解码，基线 vs 全量）")
    print(fmt_examples(base, post_full))

    plot_curves(log_entries, OUT / "figures" / "train_curves.png")


if __name__ == "__main__":
    main()
