"""
主 Agent + 并行 Subagent 编排（通用版）

教学重点：
  1. 主 agent 自己是 ReAct 循环，工具集 = 通用工具 + dispatch_subagents：
     - web_search：联网搜索一次
     - 文件工具（list_dir/read_file/write_file/delete_file，沙箱化）
     - dispatch_subagents：把任务拆成多个独立子任务派发给 subagent 并行处理
     主 agent 根据 query 自主决定用哪个——不是固定拓扑，是 LLM 自主路由
  2. 并行优势凸显：dispatch_subagents 一次派发 N 个 subagent，
     ThreadPoolExecutor 并行跑，wall-clock ≈ max(单agent时长)，
     而非 sum——这就是 subagent 并行的核心价值
  3. 每个 subagent 也是 ReAct 循环（工具 = 通用工具，不含 dispatch，不嵌套派发），
     trace 全程捕获存入 shared_state，供可视化「点节点看 ReAct 过程」

安全边界：文件操作限定 workspace/ 沙箱目录，无命令执行。
"""
import os, time, json, logging, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result
from file_tools import get_file_tools

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是任务总调度员。你有以下工具：
- web_search：联网搜索一次（参数=查询词）。用于单一事实查询
- list_dir / read_file / write_file / delete_file：沙箱内文件操作
- dispatch_subagents：派发多个子任务员并行工作（参数=用 | 分隔的多个子任务）

【关键决策原则】
- 只要任务能拆成 2 个及以上相互独立、可并行处理的子任务，
  必须用 dispatch_subagents 把各子任务分给子任务员并行处理，不要自己串行做完。
  示例："分析这3个文件各自的优缺点" → Action: dispatch_subagents
        Action Input: 分析文件a的优缺点 | 分析文件b的优缺点 | 分析文件c的优缺点
- 只有单一、不可拆分的任务才直接自己处理（搜索或文件操作）。
- 拿到各子任务结果后，综合成结构化结论。

报告要求：分要点组织，每个要点说明来源或依据，末尾给结论与注意事项。

【示例】
Question: 分析 sandbox/notes 目录下 3 个文件的要点
Thought: 这是 3 个独立文件分析，可并行处理，必须派发子任务员
Action: dispatch_subagents
Action Input: 分析 notes/a.txt 的要点 | 分析 notes/b.txt 的要点 | 分析 notes/c.txt 的要点
Observation: 并行完成：3 个子任务员...（各子任务结果）
Thought: 已收齐三个子任务结果，综合成结论
Final Answer: （分要点结论）"""


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。
    action_input: "子任务1 | 子任务2 | ..."（管道分隔）
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（eval A/B 对比用）。
    并行优势量化：wall_clock vs sum_durations。
    ⚠️ 用真实 subagent id 发 dispatch 事件（与 subagent_step 事件的 id 一致）。"""
    subtasks = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtasks:
        return "未解析出子任务"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # 构造 (sid, subagent, subtask) 三元组
    defs = []
    for topic in subtasks:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools=SUB_TOOLS,
            max_steps=6, model_tag="deepseek-chat(子)")
        defs.append((sid, sub, topic))

    # 记录派发（拓扑可视化用：主→N 个子节点）
    dispatch_info = {"subtopics": subtasks,
                     "subagent_ids": [sid for sid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    t0 = time.time()
    results = {}
    def _run_one(sid, sub, topic):
        return sid, sub.run(topic, on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    if serial:
        for sid, sub, topic in defs:
            sid, res = _run_one(sid, sub, topic)
            topic = next(t for s, _, t in defs if s == sid)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, topic): sid for sid, sub, topic in defs}
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t in defs if s == sid)
                results[sid] = (topic, res)
                shared_state["subagents"][sid] = {
                    "subtopic": topic, "trace": res["trace"],
                    "duration": res["duration"], "final_answer": res["final_answer"]}
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    shared_state.setdefault("parallel_stats", []).append({
        "n_subagents": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0})

    # 汇总文本（喂回主 agent 当 Observation，每个子结果截短避免 context 过长）
    parts = [f"【子任务: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:500]}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行完成：{len(defs)} 个子任务员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


# ── 工具注册中心：主 agent 与 subagent 从注册表组装 ─────────────────────────
COMMON_TOOLS = {
    "web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                   "联网搜索一次，参数=查询词"),
    **get_file_tools(),
}
SUB_TOOLS = dict(COMMON_TOOLS)  # subagent：搜索 + 文件，不含 dispatch（不嵌套派发）


def run_task(question: str, on_main_step: Callable = None,
             on_subagent_step: Callable = None,
             on_subagent_done: Callable = None,
             on_dispatch: Callable = None,
             serial: bool = False) -> dict:
    """执行一次通用任务。返回 {final_answer, main_trace, subagents, parallel_stats}。
    serial=True 时 subagent 串行执行（eval A/B 对比基线）。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        return _dispatch_subagents(action_input, shared_state=info,
                                   on_subagent_step=on_subagent_step,
                                   on_subagent_done=on_subagent_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial)

    MAIN_TOOLS = {
        **COMMON_TOOLS,
        "dispatch_subagents": (dispatch_tool,
                               "派发多个子任务员并行工作，参数=用 | 分隔的多个子任务"),
    }

    main = ReActLoop(
        agent_name="main",
        tools=MAIN_TOOLS,
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,
    )
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


# 兼容别名：旧调用名 run_research → run_task
run_research = run_task


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    q = "分析 workspace/notes 下 3 个文件的要点"
    r = run_task(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n结果头:\n{r['final_answer'][:200]}")
