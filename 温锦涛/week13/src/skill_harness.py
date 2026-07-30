"""
SkillHarness — 渐进式加载与执行 skills 的运行时模块

使用方式（由 Agent 调用）：
  from src.skill_harness import SkillHarness
  h = SkillHarness()
  h.list_skills()              # 查看可用 skill
  h.get_stage_manifest(name)   # 查看某个 skill 的阶段结构
  h.load_stage(name, stage)    # 按需加载某阶段内容
  h.check_prerequisites(name)  # 检查工具链
  h.resolve_path(name, rel)    # 解析 skill 内文件路径
"""

import os
import re
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent.parent / "skills"


class SkillNotFoundError(LookupError):
    """请求的 skill 不存在时抛出"""


@dataclass
class StageContent:
    content: str
    char_count: int = 0
    line_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.content)
        self.line_count = len(self.content.splitlines()) if self.content else 0


# ── Frontmatter 解析（自实现，不依赖 pyyaml）────────────────────────────────


def _parse_frontmatter(text: str) -> dict:
    """提取 SKILL.md 中 --- 之间的 YAML 块，返回有限子集的解析结果"""
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}
    return _parse_flat_yaml(m.group(1))


def _parse_flat_yaml(text: str) -> dict:
    """极简 YAML 解析器，只处理 stages 声明所需子集：
    所有块解析函数返回 (value, end_index)，end_index 是下一个未处理的行号。
    """
    result = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        m = re.match(r"^(\w[\w_-]*)\s*:\s*(.*)", line)
        if not m:
            i += 1
            continue
        key = m.group(1)
        val = m.group(2).strip()
        if not val:
            items, i = _parse_block(lines, i + 1)
            result[key] = items if items else ""
        else:
            result[key] = _parse_scalar(val)
            i += 1
    return result


def _parse_block(lines: list[str], start: int) -> tuple:
    """返回 (value, end_index)，end_index 是下一个未处理的行号"""
    if start >= len(lines):
        return None, start
    first = lines[start].strip()
    if first.startswith("- "):
        return _parse_list_block(lines, start)
    if ":" in first and not first.startswith("#"):
        return _parse_dict_block(lines, start)
    return None, start


def _parse_list_block(lines: list[str], start: int) -> tuple:
    items = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("- "):
            items.append(_parse_scalar(stripped[2:]))
            i += 1
        else:
            break
    return items, i


def _parse_dict_block(lines: list[str], start: int) -> tuple:
    result = {}
    indent = len(lines[start]) - len(lines[start].lstrip())
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        cur_indent = len(lines[i]) - len(lines[i].lstrip())
        if cur_indent < indent:
            break
        m = re.match(r"^(\w[\w_-]*)\s*:\s*(.*)", stripped)
        if not m:
            i += 1
            continue
        key_m = m.group(1)
        val_m = m.group(2).strip()
        if val_m:
            result[key_m] = _parse_scalar(val_m)
            i += 1
        else:
            sub_val, i = _parse_block(lines, i + 1)
            result[key_m] = sub_val if sub_val else ""
    return result, i


def _parse_scalar(val: str):
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [s.strip().strip("\"'") for s in inner.split(",")]
    return val.strip("\"'")


def _extract_stages(frontmatter: dict) -> dict:
    """从解析后的 frontmatter 中提取 stages 字典并验证"""
    stages_raw = frontmatter.get("stages")
    if not isinstance(stages_raw, dict):
        return {}
    valid = {}
    for stage_id, stage_def in stages_raw.items():
        if not isinstance(stage_def, dict):
            logger.warning(f"Stage '{stage_id}' 格式错误，跳过")
            continue
        if "select" not in stage_def:
            logger.warning(f"Stage '{stage_id}' 缺少 'select' 字段，跳过")
            continue
        valid[stage_id] = stage_def
    return valid


# ── 标题匹配 & 内容提取 ────────────────────────────────────────────────────


def _resolve_select(skill_dir: Path, select_val, params: dict = None) -> str:
    """按 select 定义提取内容。
    select_val 可以是字符串或列表。
    字符串：以 ## 开头 → 标题匹配，否则 → 文件路径。
    列表：递归处理每个元素，按 \n\n 拼接。
    """
    if isinstance(select_val, list):
        parts = []
        for s in select_val:
            part = _resolve_select(skill_dir, s, params)
            if part:
                parts.append(part)
        return "\n\n".join(parts)

    raw = str(select_val)

    # 动态阶段变量替换
    if params:
        for k, v in params.items():
            raw = raw.replace(f"{{{k}}}", str(v))

    if raw.startswith("## "):
        return _extract_by_heading(skill_dir / "SKILL.md", raw)

    # 文件路径
    file_path = skill_dir / raw
    if file_path.exists() and file_path.is_file():
        return file_path.read_text(encoding="utf-8").strip()

    logger.warning(f"select 路径不存在: {file_path}")
    return ""


def _extract_by_heading(md_path: Path, target: str) -> str:
    """从 Markdown 文件中按标题提取内容。
    定位首个匹配 target 的标题，从下一行收集到同级/上级标题或文件末。
    """
    if not md_path.exists():
        logger.warning(f"文件不存在: {md_path}")
        return ""
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    target_pattern = re.compile(re.escape(target) + r"\s*$")
    start_idx = -1
    for i, line in enumerate(lines):
        if target_pattern.match(line.strip()):
            start_idx = i + 1
            break
    if start_idx == -1:
        logger.warning(f"未找到标题 '{target}' 在 {md_path.name}")
        return ""

    result = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if stripped.startswith("# ") or stripped.startswith("## "):
            break
        result.append(line)
    return "\n".join(result).strip()


# ── 前置检查 ───────────────────────────────────────────────────────────────

_RUNTIME_MAP = {
    ".py":  ("python",       "python",       None),
    ".ts":  ("bun",          "bun",          "npx -y bun"),
    ".js":  ("node",         "node",         None),
    ".sh":  ("bash",         "bash",         None),
}


def _infer_runtimes(skill_dir: Path) -> list[dict]:
    """扫描 skills/{name}/scripts/ 目录，推断所需运行时"""
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.exists():
        return []
    runtimes = {}
    for f in scripts_dir.iterdir():
        if f.is_file():
            ext = f.suffix.lower()
            info = _RUNTIME_MAP.get(ext)
            if info and info[0] not in runtimes:
                runtimes[info[0]] = {
                    "tool": info[0],
                    "check": info[1],
                    "fallback": info[2],
                }
    return list(runtimes.values())


def _check_runtime_available(tool: str, fallback: str = None) -> bool:
    """运行 {tool} --version 检查可用性"""
    try:
        subprocess.run(
            [tool, "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        if fallback:
            try:
                subprocess.run(
                    fallback.split() + ["--version"],
                    capture_output=True,
                    timeout=10,
                )
                return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return False


# ── SkillHarness ───────────────────────────────────────────────────────────


class SkillHarness:
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self._skills_dir = skills_dir
        self._skill_map: dict[str, Path] = {}
        self._refresh()

    def _refresh(self):
        """扫描 skills/ 目录，建立名称 → 目录的映射"""
        self._skill_map = {}
        if not self._skills_dir.exists():
            return
        for entry in self._skills_dir.iterdir():
            if entry.is_dir():
                skill_md = entry / "SKILL.md"
                if skill_md.exists():
                    self._skill_map[entry.name] = entry

    def _get_skill_path(self, name: str) -> Path:
        path = self._skill_map.get(name)
        if not path:
            raise SkillNotFoundError(f"Skill '{name}' 不存在（路径: {self._skills_dir / name / 'SKILL.md'}）")
        return path

    # ── 发现阶段 ────────────────────────────────────────────────────────

    def list_skills(self) -> list[dict]:
        """返回所有已发现 skill 的元数据"""
        result = []
        for name, path in self._skill_map.items():
            skill_md = path / "SKILL.md"
            text = skill_md.read_text(encoding="utf-8")
            fm = _parse_frontmatter(text)
            stages = _extract_stages(fm)
            result.append({
                "name": name,
                "description": fm.get("description", ""),
                "stages": list(stages.keys()),
                "path": str(path),
            })
        return result

    def get_stage_manifest(self, name: str) -> dict:
        """返回 skill 的完整阶段定义"""
        path = self._get_skill_path(name)
        text = (path / "SKILL.md").read_text(encoding="utf-8")
        fm = _parse_frontmatter(text)
        stages = _extract_stages(fm)
        if not stages:
            stages = {"full": {"select": str(path / "SKILL.md")}}
        return {
            "name": name,
            "stages": stages,
        }

    # ── 加载阶段 ────────────────────────────────────────────────────────

    def load_stage(self, name: str, stage_id: str,
                   params: dict = None) -> StageContent:
        """加载 skill 的某个阶段，返回内容"""
        manifest = self.get_stage_manifest(name)
        stages = manifest["stages"]
        if stage_id not in stages:
            raise ValueError(
                f"Skill '{name}' 无阶段 '{stage_id}'，可用: {list(stages)}"
            )
        stage_def = stages[stage_id]
        skill_dir = self._get_skill_path(name)

        # 隐式 full 阶段：直接读文件
        if stage_id == "full" and stage_def.get("select", "").endswith("SKILL.md"):
            content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
            return StageContent(content=content)

        content = _resolve_select(skill_dir, stage_def["select"], params)
        return StageContent(content=content)

    # ── 前置检查 ────────────────────────────────────────────────────────

    def check_prerequisites(self, name: str) -> list[dict]:
        """检查 skill 的工具链是否就绪"""
        skill_dir = self._get_skill_path(name)
        runtimes = _infer_runtimes(skill_dir)
        results = []
        for rt in runtimes:
            available = _check_runtime_available(rt["check"], rt.get("fallback"))
            entry = {"tool": rt["tool"], "available": available}
            if not available:
                hints = {
                    "python": "安装 Python: https://python.org",
                    "bun": "安装 Bun: https://bun.sh",
                    "node": "安装 Node.js: https://nodejs.org",
                    "bash": "bash 通常已预装",
                }
                entry["hint"] = hints.get(rt["tool"], f"请安装 {rt['tool']}")
            results.append(entry)
        return results

    # ── 路径解析 ────────────────────────────────────────────────────────

    def resolve_path(self, name: str, relative: str) -> str:
        """将 skill 内相对路径转为绝对路径"""
        skill_dir = self._get_skill_path(name)
        return str((skill_dir / relative).resolve())

    def get_skill_dir(self, name: str) -> str:
        """返回 skill 目录的绝对路径"""
        return str(self._get_skill_path(name).resolve())
