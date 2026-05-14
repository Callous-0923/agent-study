"""
章节 .py → 可读 .html 转换器

用法:
  python build_html.py chapter_00_overview/00_course_overview.py
  python build_html.py chapter_01_fundamentals/01_hello_agent.py

生成的 HTML 包含:
  - 讲义内容（模块 docstring → 格式化排版）
  - 代码块（语法高亮）
  - 上一章/下一章 导航
  - 移动端适配
"""

import re
import sys
import os

BASE_URL = "https://callous-0923.github.io/agent-study"

CHAPTERS = {
    0:  ("chapter_00_overview",      "00_course_overview.html",           "课程概览与环境搭建"),
    1:  ("chapter_01_fundamentals",   "01_hello_agent.html",              "第一个 Agent — 裸写 ReAct 循环"),
    2:  ("chapter_02_components",    "02_agent_components.html",         "Agent 核心组件 — 规划器·记忆·工具设计"),
    3:  ("chapter_03_types",         "03_agent_types.html",              "Agent 类型分类 — ReAct·Plan-Execute·Reflexion"),
    4:  ("chapter_04_frameworks",    "04_frameworks.html",               "主流框架实战 — LangChain·LangGraph"),
    5:  ("chapter_05_multi_agent",   "05_multi_agent.html",              "多智能体系统 — Multi-Agent 协作"),
    6:  ("chapter_06_evaluation",    "06_evaluation.html",               "评估与测试 — 评测框架·生产 Checklist"),
    7:  ("chapter_07_interview",     "07_interview_prep.html",           "求职面试准备 — 20 道高频题·项目指南"),
    8:  ("chapter_08_claude_code",   "08_claude_code_architecture.html", "Claude Code 架构 — nO·h2A·压缩·SubAgent"),
    9:  ("chapter_09_rag_deepdive",  "09_rag_deepdive.html",             "RAG 深度剖析 — Naive→Advanced→GraphRAG→Agentic RAG"),
    10: ("chapter_10_mcp",           "10_mcp_deepdive.html",             "MCP 协议详解 — JSON-RPC·原语·能力协商"),
    11: ("chapter_11_tool_calling",  "11_tool_calling_deepdive.html",    "Tool Calling 底层 — OpenAI vs Anthropic"),
    12: ("chapter_12_infrastructure","12_infrastructure.html",           "Agent 生产基础设施 — OpenClaw·Harness·Checklist"),
    13: ("chapter_13_fastapi",       "13_fastapi_agent_service.html",    "FastAPI 服务化 — REST·SSE·WebSocket"),
    14: ("chapter_14_sqlite",        "14_sqlite_agent_storage.html",     "SQLite 持久化 — 5 表 Schema·WAL·审计查询"),
    15: ("chapter_15_a2a",           "15_a2a_protocol.html",             "Google A2A 协议 — AgentCard·Task·Artifact"),
    16: ("chapter_16_memgpt",        "16_memgpt_letta.html",             "MemGPT/Letta 记忆 — Core Memory·Heartbeat·Sleep-Time"),
    17: ("chapter_17_computer_use",  "17_computer_use.html",             "Computer Use — Screenshot-Action Loop·安全沙箱"),
    18: ("chapter_18_security",      "18_agent_security.html",           "Agent 安全与护栏 — Prompt Injection·权限分级"),
    19: ("chapter_19_workflow_patterns","19_workflow_patterns.html",     "Agentic Workflow 设计模式 — Reflection·Routing·Orchestrator"),
    20: ("chapter_20_context_engineering","20_context_engineering.html", "Context Engineering — Context Rot·预算管理·XML Prompt"),
    21: ("chapter_21_streaming",     "21_streaming_architecture.html",   "Streaming & 实时架构 — EventBus·动态中断·背压"),
    22: ("chapter_22_dspy",          "22_dspy.html",                     "DSPy 自动优化 — Signature·Module·Optimizer"),
    23: ("chapter_23_code_agents",   "23_code_agents.html",              "代码 Agent 架构横评 — CodeAct·ACI·SWE-bench"),
    24: ("chapter_24_observability", "24_observability.html",            "Agent 可观测性 — Tracing·LangSmith vs LangFuse"),
    25: ("chapter_25_vectordb",      "25_vectordb.html",                 "向量数据库选型 — Chroma·Pinecone·Milvus·Qdrant"),
    26: ("chapter_26_model_routing", "26_model_routing.html",            "模型路由策略 — Threshold·Cascade·Semantic·Cost-Aware"),
    27: ("chapter_27_prompt_eng",    "27_prompt_engineering.html",       "Agent Prompt 工程 — System Prompt 模板·工具描述"),
    28: ("chapter_28_cache",         "28_cache.html",                    "语义缓存与 Token 优化 — 三级缓存·预算管理"),
}


CSS = r"""
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
  background: #f8f9fa; color: #1a1a2e; line-height: 1.8;
}
.container { max-width: 900px; margin: 0 auto; padding: 40px 24px 80px; }

/* 标题 */
.hero {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white; padding: 48px 32px; border-radius: 12px; margin-bottom: 36px;
}
.hero h1 { font-size: 2em; margin-bottom: 8px; }
.hero .subtitle { opacity: 0.9; font-size: 1.05em; }

/* 讲义内容 */
.lecture { background: white; border-radius: 10px; padding: 36px 32px;
  box-shadow: 0 2px 12px rgba(0,0,0,.06); margin-bottom: 32px; }
.lecture h2 {
  font-size: 1.35em; color: #667eea; margin: 28px 0 12px;
  padding-bottom: 6px; border-bottom: 2px solid #e8e8ff;
}
.lecture h3 { font-size: 1.1em; color: #444; margin: 20px 0 8px; }
.lecture p { margin: 10px 0; }
.lecture ul, .lecture ol { margin: 8px 0 8px 24px; }
.lecture li { margin: 4px 0; }
.lecture pre {
  background: #1e1e2e; color: #cdd6f4; border-radius: 8px;
  padding: 18px 22px; overflow-x: auto; font-size: .88em;
  line-height: 1.55; margin: 14px 0;
}
.lecture code { font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace; }
.lecture p code, .lecture li code {
  background: #eee8ff; padding: 1px 6px; border-radius: 4px;
  font-size: .92em; color: #6c3cc0;
}
.lecture strong { color: #333; }
.lecture em { color: #555; }
.lecture blockquote {
  border-left: 4px solid #667eea; padding: 8px 16px;
  margin: 14px 0; background: #f4f4ff; border-radius: 0 8px 8px 0;
  color: #555;
}
.lecture table {
  border-collapse: collapse; width: 100%; margin: 14px 0;
  font-size: .92em;
}
.lecture th, .lecture td {
  border: 1px solid #e0e0e0; padding: 8px 12px; text-align: left;
}
.lecture th { background: #f0f0ff; font-weight: 600; }
.lecture hr { border: none; border-top: 1px solid #e0e0e0; margin: 24px 0; }

/* 目标/考点标签 */
.tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }
.tag {
  display: inline-block; padding: 4px 12px; border-radius: 20px;
  font-size: .85em; font-weight: 500;
}
.tag-goal { background: #e8f5e9; color: #2e7d32; }
.tag-interview { background: #fff3e0; color: #e65100; }

/* 代码块区域 */
.code-section { margin-top: 24px; }
.code-section summary {
  font-size: 1.1em; font-weight: 600; cursor: pointer;
  padding: 10px 16px; background: #1e1e2e; color: #cdd6f4;
  border-radius: 8px 8px 0 0; user-select: none;
}
.code-section pre {
  margin: 0; border-radius: 0 0 8px 8px; max-height: 70vh;
  overflow-y: auto;
}
.code-block {
  background: #1e1e2e; color: #cdd6f4; border-radius: 0 0 8px 8px;
  padding: 20px 24px; overflow-x: auto; font-size: .85em;
  line-height: 1.6; max-height: 70vh; overflow-y: auto;
}

/* 移动端 */
@media (max-width: 640px) {
  .container { padding: 16px 12px 60px; }
  .hero { padding: 28px 18px; }
  .hero h1 { font-size: 1.4em; }
  .lecture { padding: 20px 16px; }
}

/* 打印 */
@media print {
  body { background: white; }
  .lecture { box-shadow: none; }
  .code-block { background: #f5f5f5; color: #222; }
}

/* 上/下章导航 */
.nav-bar {
  display: flex; justify-content: space-between; align-items: center;
  margin: 32px 0; gap: 16px;
}
.nav-link {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 12px 22px; border-radius: 10px; text-decoration: none;
  font-weight: 600; font-size: .95em; transition: all .2s;
}
.nav-prev { background: #f0f0ff; color: #667eea; }
.nav-prev:hover { background: #e0e0ff; }
.nav-next { background: #667eea; color: white; }
.nav-next:hover { background: #5560d8; }
.nav-disabled { background: #eee; color: #999; cursor: not-allowed; pointer-events: none; }
.nav-home { background: #f5f5f5; color: #555; }
.nav-home:hover { background: #e0e0e0; }

/* 面包屑 */
.breadcrumb {
  font-size: .88em; color: #888; margin-bottom: 20px;
}
.breadcrumb a { color: #667eea; text-decoration: none; }
.breadcrumb a:hover { text-decoration: underline; }
"""

HEADER = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<div class="container">
"""

FOOTER = """</div></body></html>"""


def _make_nav(ch_num: int) -> str:
    """生成上/下章导航 HTML。"""
    prev_ch = ch_num - 1
    next_ch = ch_num + 1

    parts = ['<div class="nav-bar">']

    # 上一章
    if prev_ch in CHAPTERS:
        dir_name, file_name, ch_title = CHAPTERS[prev_ch]
        url = f"{BASE_URL}/{dir_name}/{file_name}"
        label = f"← 第{prev_ch}章"
        parts.append(f'<a class="nav-link nav-prev" href="{url}" title="{ch_title}">{label}</a>')
    else:
        parts.append('<span class="nav-link nav-disabled">← 已是第一章</span>')

    # 主页
    parts.append(f'<a class="nav-link nav-home" href="{BASE_URL}/chapter_00_overview/00_course_overview.html">📖 课程主页</a>')

    # 下一章
    if next_ch in CHAPTERS:
        dir_name, file_name, ch_title = CHAPTERS[next_ch]
        url = f"{BASE_URL}/{dir_name}/{file_name}"
        label = f"第{next_ch}章 →"
        parts.append(f'<a class="nav-link nav-next" href="{url}" title="{ch_title}">{label}</a>')
    else:
        parts.append('<span class="nav-link nav-disabled">已是最后一章 →</span>')

    parts.append('</div>')
    return "\n".join(parts)


def extract_sections(filepath: str) -> tuple[str, list[tuple[str, str]], str]:
    # 提取 (讲义, 对应代码) 的交替段 + 完整源码
    # 结构: [讲义1] [代码1] [讲义2] [代码2] ... [最终代码]
    # 返回 (title, [(lecture, code), ...], full_code)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    segments = []  # [(start, end, is_lecture)]
    pos = 0
    while True:
        start = content.find('"""', pos)
        if start == -1:
            break
        # 判断是否行首
        line_start = content.rfind('\n', 0, start)
        line_start = 0 if line_start == -1 else line_start + 1
        prefix = content[line_start:start]
        if prefix != "":
            pos = start + 3
            continue
        inner_start = start + 3
        end = content.find('"""', inner_start)
        if end == -1:
            break
        end_line_start = content.rfind('\n', 0, end)
        end_line_start = 0 if end_line_start == -1 else end_line_start + 1
        end_prefix = content[end_line_start:end]
        if end_prefix != "":
            pos = end + 3
            continue
        doc = content[inner_start:end].strip()
        if doc:
            segments.append((start, end + 3, doc))
        pos = end + 3

    # 组装为讲义+代码交替列表
    sections = []
    for i, (s_start, s_end, doc) in enumerate(segments):
        next_start = segments[i + 1][0] if i + 1 < len(segments) else len(content)
        code = content[s_end:next_start].strip()
        sections.append((doc, code))

    # 标题
    title = "未命名章节"
    if sections:
        first_line = sections[0][0].split("\n")[0].strip()
        title = re.sub(r'^第\d+[章节]：?', '', first_line).strip()
    if not title:
        title = os.path.basename(filepath).replace(".py", "")

    return title, sections, content


def build_html(filepath: str, output_path: str = None):
    # 构建完整 HTML 文件（讲义+代码交替显示）
    title, sections, full_code = extract_sections(filepath)

    ch_num = 0
    m = re.search(r'chapter_(\d+)', filepath)
    if m:
        ch_num = int(m.group(1))

    html = HEADER.format(
        title=f"第{ch_num}章：{title} — AI Agent 全栈课程",
        css=CSS,
    )

    # 面包屑 + 导航 + hero
    dir_name, file_name, ch_title = CHAPTERS.get(ch_num, ("", "", ""))
    html += '<div class="breadcrumb">'
    html += f'<a href="{BASE_URL}/chapter_00_overview/00_course_overview.html">📖 AI Agent 全栈课程</a>'
    html += f' &raquo; <strong>第{ch_num}章 {ch_title}</strong>'
    html += '</div>'
    html += _make_nav(ch_num)
    html += f'<div class="hero"><h1>第{ch_num}章 {title}</h1>'
    html += f'<p class="subtitle">📖 AI Agent 全栈学习课程 · 可运行讲义</p></div>'

    # 讲义+代码交替渲染
    for i, (lecture_text, code_text) in enumerate(sections):
        # 讲义部分
        lecture_html = parse_lecture(lecture_text)
        html += f'<div class="lecture">{lecture_html}</div>'

        # 对应代码部分（如果有代码）
        if code_text:
            highlighted = highlight_python(code_text)
            lines = code_text.count('\n') + 1
            html += f'<div class="code-section">'
            html += f'<details><summary>💻 代码 ({lines} 行)</summary>'
            html += f'<div class="code-block"><pre>{highlighted}</pre></div>'
            html += f'</details></div>'

    # 完整源码（底部折叠）
    total_lines = full_code.count('\n') + 1
    highlighted_full = highlight_python(full_code)
    html += '<hr style="margin:36px 0">'
    html += f'<div class="code-section"><details>'
    html += f'<summary>📦 完整源代码 ({total_lines} 行)</summary>'
    html += f'<div class="code-block"><pre>{highlighted_full}</pre></div>'
    html += f'</details></div>'

    html += _make_nav(ch_num)
    html += FOOTER

    if output_path is None:
        base = os.path.splitext(os.path.basename(filepath))[0]
        output_path = os.path.join(os.path.dirname(filepath), f"{base}.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def parse_lecture(docstring: str) -> str:
    """
    处理:
      - ### / ## → h2/h3
      - ━━ 分隔线 → <hr>
      - ●/• 列表 → <ul>
      - 有序列表 → <ol>
      - 代码块（缩进4空格或```）→ <pre>
      - 粗体 **text** → <strong>
      - 表格 → <table> (简单检测 | 分隔)
      - ┌─┐ 等 box-drawing 字符 → <pre> 块
      - 表情符号保留
    """
    lines = docstring.split("\n")
    output = []
    i = 0
    in_code_block = False
    code_buffer = []
    in_box = False
    box_buffer = []
    in_table = False
    table_buffer = []
    goals = []
    interview_points = []

    def flush_code():
        nonlocal in_code_block, code_buffer
        if code_buffer:
            output.append("<pre>" + "\n".join(code_buffer) + "</pre>")
            code_buffer = []
            in_code_block = False

    def flush_box():
        nonlocal in_box, box_buffer
        if box_buffer:
            output.append("<pre>" + "\n".join(box_buffer) + "</pre>")
            box_buffer = []
            in_box = False

    def flush_table():
        nonlocal in_table, table_buffer
        if table_buffer:
            # 检测表格行数
            rows = []
            for tline in table_buffer:
                tline = tline.strip("│").strip()
                cells = [c.strip() for c in tline.split("│")]
                rows.append(cells)
            if len(rows) >= 2 and all(r for r in rows):
                html = "<table>"
                for ri, row in enumerate(rows):
                    tag = "th" if ri == 0 else "td"
                    html += "<tr>"
                    for cell in row:
                        html += f"<{tag}>{cell}</{tag}>"
                    html += "</tr>"
                html += "</table>"
                output.append(html)
            else:
                output.append("<pre>" + "\n".join(table_buffer) + "</pre>")
            table_buffer = []
            in_table = False

    while i < len(lines):
        line = lines[i]

        # box-drawing 块检测
        if any(c in line for c in "┌└├│┐┘┤┬┴"):
            flush_code()
            flush_table()
            if not in_box:
                in_box = True
            box_buffer.append(line)
            i += 1
            continue
        elif in_box:
            flush_box()

        # 表格检测（含 │ 且非 box 行）
        if "│" in line and not any(c in line for c in "┌└├┐┘┤"):
            flush_code()
            flush_box()
            if not in_table:
                in_table = True
            table_buffer.append(line)
            i += 1
            continue
        elif in_table:
            flush_table()

        # 代码块 ```
        if line.strip().startswith("```"):
            if in_code_block:
                flush_code()
            else:
                flush_table()
                flush_box()
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_buffer.append(line)
            i += 1
            continue

        # 缩进代码块（4 空格开头的行）
        if line.startswith("    ") and not line.strip().startswith("-") and not line.strip().startswith("1.") and not line.strip().startswith("•"):
            # 检测连续缩进
            flush_table()
            flush_box()
            cb = [line[4:]]
            j = i + 1
            while j < len(lines) and lines[j].startswith("    "):
                cb.append(lines[j][4:])
                j += 1
            output.append("<pre>" + "\n".join(cb) + "</pre>")
            i = j
            continue

        flush_code()
        flush_box()
        flush_table()

        stripped = line.strip()

        # 目标提取
        if stripped.startswith("📌 本章目标") or stripped.startswith("📌 本章结构"):
            # 收集后续的编号项
            i += 1
            while i < len(lines) and (
                re.match(r'^\s+\d+\.', lines[i]) or
                re.match(r'^\s+\d+\.\s', lines[i])
            ):
                goals.append(lines[i].strip())
                i += 1
            continue

        if stripped.startswith("📌 面试高频点"):
            i += 1
            while i < len(lines) and lines[i].strip().startswith("-"):
                interview_points.append(lines[i].strip()[1:].strip())
                i += 1
            continue

        # 空行
        if not stripped:
            output.append("<p>&nbsp;</p>")
            i += 1
            continue

        # 章节标题 ━━
        if stripped.startswith("━━") or all(c in "━" for c in stripped if c != " "):
            i += 1
            continue

        # 小节标识 (如 "0.1 Agent 是什么？—— 用类比建立直觉")
        section_match = re.match(r'^(\d+\.\d+)\s+(.+)', stripped)
        if section_match:
            output.append(f'<h2>{stripped}</h2>')
            i += 1
            continue

        # ## / ### 风格标题
        if stripped.startswith("### "):
            output.append(f'<h3>{stripped[4:]}</h3>')
            i += 1
            continue
        if stripped.startswith("## "):
            output.append(f'<h2>{stripped[3:]}</h2>')
            i += 1
            continue

        # 粗体
        stripped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)

        # 行内代码
        stripped = re.sub(r'`([^`]+)`', r'<code>\1</code>', stripped)

        # 列表项
        if re.match(r'^[-•]\s', stripped):
            output.append(f'<li>{stripped[1:].strip()}</li>')
            i += 1
            continue

        # 有序列表
        ol_match = re.match(r'^(\d+)\.\s+(.+)', stripped)
        if ol_match:
            output.append(f'<li>{ol_match.group(2)}</li>')
            i += 1
            continue

        # 引用
        if stripped.startswith(">"):
            output.append(f'<blockquote>{stripped[1:].strip()}</blockquote>')
            i += 1
            continue

        # 分隔线
        if stripped == "---":
            output.append("<hr>")
            i += 1
            continue

        # 普通段落
        output.append(f"<p>{stripped}</p>")
        i += 1

    # 清理残余
    flush_code()
    flush_box()
    flush_table()

    # 组装
    result = ""

    # 目标标签行
    if goals or interview_points:
        result += '<div class="tag-row">'
        for g in goals:
            result += f'<span class="tag tag-goal">{g}</span>'
        for ip in interview_points:
            result += f'<span class="tag tag-interview">🎤 {ip}</span>'
        result += '</div>'

    result += "\n".join(output)
    return result


def highlight_python(code: str) -> str:
    """Python 语法高亮（简单版）。"""
    # 转义 HTML
    code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 注释
    code = re.sub(r'(<span class="h">)?(#[^\n]*)(</span>)?',
                  r'<span class="h">\2</span>', code)

    # 字符串
    code = re.sub(r'("""[\s\S]*?""")', r'<span class="s">\1</span>', code)
    code = re.sub(r"('''[\s\S]*?''')", r'<span class="s">\1</span>', code)
    code = re.sub(r'"([^"\\\n]*(\\.[^"\\\n]*)*)"', r'<span class="s">"\1"</span>', code)
    code = re.sub(r"'([^'\\\n]*(\\.[^'\\\n]*)*)'", r"<span class='s'>'\1'</span>", code)

    # 关键字
    kw = r'\b(def|class|import|from|return|if|else|elif|for|while|try|except|' \
         r'with|as|in|not|and|or|is|None|True|False|async|await|yield|' \
         r'raise|break|continue|pass|lambda|global|nonlocal)\b'
    code = re.sub(kw, r'<span class="k">\1</span>', code)

    # 装饰器
    code = re.sub(r'(<span class="h">)?(@\w+)', r'<span class="d">\2</span>', code)

    # 数字
    code = re.sub(r'\b(\d+\.?\d*)\b', r'<span class="n">\1</span>', code)

    # f-string 前缀
    code = code.replace('<span class="s">f"', '<span class="s">f"')
    code = code.replace("<span class='s'>f'", "<span class='s'>f'")

    return code


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python build_html.py <章节.py>")
        sys.exit(1)

    for path in sys.argv[1:]:
        out = build_html(path)
        print(f"✅ {path} → {out}")
