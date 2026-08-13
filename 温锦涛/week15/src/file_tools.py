"""
沙箱化文件工具（通用 subagent 派发系统用）

教学重点：
  1. 工具不抛异常，失败返回可读错误字符串 —— ReAct 循环拿到 Observation 继续兜底
  2. 安全边界：所有文件操作限定在 workspace/ 沙箱目录内，
     路径经 realpath 规范化后必须落在沙箱根内，越界拒绝（防路径穿越）
  3. 读文件内容截断，避免撑爆 LLM context

安全模型：
  - 允许 读 / 写 / 删 / 列目录，限定沙箱目录
  - 不提供任何命令执行能力（run_command 显式排除）
"""
import os, re, json, logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 沙箱根：相对仓库根（本文件在 src/ 下）
BASE_DIR = Path(__file__).parent.parent
SANDBOX = BASE_DIR / "workspace"

# 单次读取返回的内容上限（防 context 撑爆）
READ_LIMIT = 600
# 单次列目录条数上限
LIST_LIMIT = 50
# 文件名/路径清洗（限制非法字符，教学演示用）
_FORBIDDEN = re.compile(r'[\x00-\x1f<>:"|?*]')


def _resolve(path: str = "") -> tuple[str, str]:
    """把 LLM 给的路径解析为沙箱内绝对路径。
    返回 (abs_path, err)：err 非空表示越界/非法，abs_path 为空。
    """
    path = (path or "").strip()
    # 空 → 沙箱根
    if not path:
        return str(SANDBOX), ""
    # 去掉开头的绝对/上级指示（只允许相对路径）
    p = path.replace("\\", "/").lstrip("/")
    if _FORBIDDEN.search(p):
        return "", f"路径含非法字符: {path!r}"
    # 拼接并规范化，必须落在沙箱内
    candidate = (SANDBOX / p)
    abs_p = os.path.realpath(str(candidate))
    sandbox_real = os.path.realpath(str(SANDBOX))
    if abs_p != sandbox_real and not abs_p.startswith(sandbox_real + os.sep):
        return "", f"路径越出沙箱目录，拒绝: {path!r}"
    return abs_p, ""


def list_dir(path: str = "", *, recursive: bool = False) -> str:
    """列出目录内容。path 缺省列出沙箱根。返回文本列表。"""
    abs_p, err = _resolve(path)
    if err:
        return err
    try:
        entries = sorted(os.listdir(abs_p))
    except FileNotFoundError:
        return f"目录不存在: {abs_p}"
    except NotADirectoryError:
        return f"不是目录: {abs_p}"
    except PermissionError:
        return f"无权限访问: {abs_p}"
    lines = []
    for name in entries[:LIST_LIMIT]:
        full = os.path.join(abs_p, name)
        kind = "dir" if os.path.isdir(full) else "file"
        size = ""
        if os.path.isfile(full):
            try:
                size = f" ({os.path.getsize(full)}B)"
            except OSError:
                pass
        lines.append(f"  [{kind}] {name}{size}")
    if len(entries) > LIST_LIMIT:
        lines.append(f"  … 还有 {len(entries) - LIST_LIMIT} 项未显示")
    if not lines:
        return "(空目录)"
    header = f"目录 {abs_p}:"
    return header + "\n" + "\n".join(lines)


def read_file(path: str, *, limit: int = READ_LIMIT) -> str:
    """读取文件内容，截断到 limit 字符。失败返回错误字符串。"""
    abs_p, err = _resolve(path)
    if err:
        return err
    try:
        if not os.path.isfile(abs_p):
            return f"文件不存在: {abs_p}"
        with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except PermissionError:
        return f"无权限读取: {abs_p}"
    except OSError as e:
        return f"读取失败: {type(e).__name__}: {str(e)[:100]}"
    if len(content) > limit:
        content = content[:limit] + f"\n…(内容已截断, 共 {limit} 字)"
    return content


def write_file(path: str, content: str = "") -> str:
    """写入/覆盖文件，自动创建父目录。返回确认或错误。"""
    abs_p, err = _resolve(path)
    if err:
        return err
    try:
        parent = os.path.dirname(abs_p)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(abs_p, "w", encoding="utf-8") as f:
            f.write(content or "")
    except PermissionError:
        return f"无权限写入: {abs_p}"
    except OSError as e:
        return f"写入失败: {type(e).__name__}: {str(e)[:100]}"
    return f"已写入 {len(content)} 字符到 {abs_p}"


def delete_file(path: str) -> str:
    """删除文件。默认仅文件，目录需显式 recursive=True。"""
    abs_p, err = _resolve(path)
    if err:
        return err
    try:
        if os.path.isdir(abs_p):
            return f"是目录，请用文件名（教学演示不开放删目录）"
        if not os.path.exists(abs_p):
            return f"文件不存在: {abs_p}"
        os.remove(abs_p)
    except PermissionError:
        return f"无权限删除: {abs_p}"
    except OSError as e:
        return f"删除失败: {type(e).__name__}: {str(e)[:100]}"
    return f"已删除 {abs_p}"


def get_file_tools() -> dict:
    """返回文件工具注册表 {name: (fn, description)}。"""
    return {
        "list_dir": (lambda path="", **_: list_dir(path),
                      "列出目录内容，参数=路径（可省略，缺省列沙箱根）"),
        "read_file": (lambda path, **_: read_file(path),
                      "读取文件内容，参数=文件路径"),
        "write_file": (lambda path, content="", **_: write_file(path, content),
                       "写入/覆盖文件，参数='路径|内容'（用 | 分隔）"),
        "delete_file": (lambda path, **_: delete_file(path),
                        "删除文件，参数=文件路径"),
    }


# ── 自测：读写删 + 越界拒绝 ────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    print("=== 沙箱根 ===")
    print(list_dir())

    print("\n=== 写入 ===")
    print(write_file("notes/hello.txt", "你好，通用 subagent！"))

    print("\n=== 读取 ===")
    print(read_file("notes/hello.txt"))

    print("\n=== 列出 ===")
    print(list_dir("notes"))

    print("\n=== 越界拒绝 ===")
    print(read_file("../../etc/passwd"))
    print(write_file("../outside.txt", "hack"))

    print("\n=== 删除 ===")
    print(delete_file("notes/hello.txt"))
    print(list_dir("notes"))
