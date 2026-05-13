"""
第0章：AI Agent 课程概览与前置知识
===================================

📌 本章目标：
  1. 理解「什么是 AI Agent」- 建立直觉认知
  2. 掌握 Agent 的核心组成公式：Agent = LLM + 规划 + 记忆 + 工具
  3. 了解 Agent 开发的完整技术栈和 18 章学习路线
  4. 确认学习本课程所需的前置知识

📌 本章结构：
  0.1 Agent 是什么？（5分钟理解）
  0.2 为什么 2024-2026 年是 Agent 爆发期？
  0.3 Agent 核心公式：Agent = LLM + 规划 + 记忆 + 工具
  0.4 全套学习路线图（18章四层递进）
  0.5 前置知识自查清单
  0.6 环境搭建（一键运行）

---

0.1 Agent 是什么？—— 用类比建立直觉
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ 普通 LLM（如 ChatGPT 基础版）：
   用户：「北京明天天气怎么样？」
   LLM：「抱歉，我无法获取实时数据。」  ← 只会"说"，不会"做"

✅ AI Agent（如 AutoGPT / LangChain Agent）：
   用户：「北京明天天气怎么样？」
   Agent：
     1. 思考：我需要调用天气 API
     2. 行动：调用 get_weather("北京") → 获取 JSON 数据
     3. 观察：返回 {"明天": "晴，25°C"}
     4. 回答：「北京明天晴天，气温 25°C」  ← 能"说"也能"做"

核心区别：
  - LLM 是「大脑」（会思考）
  - Agent 是「大脑 + 手」（会思考 + 会行动）

类比：
  把 LLM 想象成一个被困在房间里的人，只能通过文字交流。
  Agent 就是给他开了门，让他可以走出去使用工具、查阅资料、执行操作。

---

0.2 为什么 Agent 是 2024-2026 年最热门方向？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

关键时间线：

  ── 2023: LLM 之年 ──
    • 2023.03: AutoGPT 开源，Star 数一周破 10 万
    • 2023.06: OpenAI 发布 Function Calling
    • 2023.10: LangChain 发布 LangGraph（Agent 编排框架）
    • 2023.12: MemGPT 论文发布（LLM as Operating System）

  ── 2024: RAG 与 Agent 元年 ──
    • 2024.05: OpenAI 发布 GPT-4o，原生支持工具调用
    • 2024.10: Anthropic 发布 Computer Use（Claude 操控电脑）
    • 2024.11: Anthropic 发布 MCP 协议（Agent ↔ 工具的 USB-C）

  ── 2025: Agent 爆发之年 ──
    • 2025.01: DeepSeek-R1 发布，推理能力大幅提升
    • 2025.02: Claude Code 正式发布（工业级编码 Agent 标杆）
    • 2025.04: Google 联合 50+ 企业发布 A2A 协议（Agent ↔ Agent）
    • 2025.09: Anthropic 发布 Claude Agent SDK（通用 Agent 框架）
    • 2025.10: Anthropic Computer Use 正式发布（production-ready）
    • 2025.11: OpenClaw (原 Clawdbot) 开源，145K+ GitHub Stars
    • 2025.12: OpenAI 发布 CUA (Computer Using Agent)

  ── 2026: Agent 基础设施成熟 ──
    • 2026.02: Harness 在 Harness-Bench 上达到 100% 评分
    • 2026.03: Letta 发布 Filesystem Memory 方案（LoCoMo 74.0%）
    • 2026: MCP / A2A 成为行业标准，Agent 中间件生态涌现

就业信号（2025-2026）：
  - "AI Agent 工程师" 岗位同比增长 300%+
  - 大厂（字节/腾讯/阿里/Google/Meta）均设立 Agent 专项团队
  - 硅谷 VC 投资方向全面转向 Agent 赛道
  - Agent 工程师薪资范围：一线城市 25K-80K（视经验和公司）

---

0.3 Agent 核心公式
━━━━━━━━━━━━━━━━━━

  Agent = LLM + 规划(Planning) + 记忆(Memory) + 工具(Tools)

  ┌─────────────────────────────────────────────┐
  │                  AI Agent                    │
  │  ┌─────────┐  ┌────────┐  ┌──────────────┐ │
  │  │   LLM   │  │ 规划器  │  │   记忆系统    │ │
  │  │ (大脑)  │  │Planning│  │   Memory     │ │
  │  └────┬────┘  └───┬────┘  └──────┬───────┘ │
  │       │           │              │          │
  │       └───────────┼──────────────┘          │
  │                   │                         │
  │            ┌──────┴──────┐                  │
  │            │   工具调用    │                  │
  │            │   Tools     │                  │
  │            │ 搜索/API/代码│                  │
  │            └─────────────┘                  │
  └─────────────────────────────────────────────┘

  Agent 执行循环（核心，面试高频！）：
    1. 感知(Perceive)：接收用户输入和上一次行动的结果
    2. 思考(Think)：LLM 分析当前状态，决定下一步
    3. 行动(Act)：调用工具或生成最终回答
    4. 观察(Observe)：获取行动结果
    → 回到步骤 1，直到任务完成

  这个循环被称为「ReAct 循环」(Reasoning + Acting)，
  是 99% 的 Agent 框架的底层逻辑。


0.4 全套学习路线图（18章四层递进）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本课程按「理论 → 框架 → 深度 → 工程」四层递进组织：

  ┌─────────────────────────────────────────────────────────────┐
  │ 第1层：Agent 理论基础（Ch1-3）                                │
  │ ┌──────────────┬──────────────────────────────────────────┐ │
  │ │ Ch1 基础概念  │ Agent 定义、LLM 调用、手写第一个 Agent    │ │
  │ │ Ch2 核心组件  │ 规划器 + 记忆系统 + 工具设计黄金法则       │ │
  │ │ Ch3 类型分类  │ ReAct / Plan-Execute / Reflexion 对比    │ │
  │ └──────────────┴──────────────────────────────────────────┘ │
  ├─────────────────────────────────────────────────────────────┤
  │ 第2层：工程实践与框架（Ch4-7）                                │
  │ ┌──────────────┬──────────────────────────────────────────┐ │
  │ │ Ch4 主流框架  │ LangChain Agent + LangGraph 状态机实战    │ │
  │ │ Ch5 多智能体  │ 协作式/分层式 Multi-Agent 架构            │ │
  │ │ Ch6 评估测试  │ 评测框架 + 测试策略 + 生产 Checklist       │ │
  │ │ Ch7 求职面试  │ 20道高频面试题 + 项目指南 + 面试流程       │ │
  │ └──────────────┴──────────────────────────────────────────┘ │
  ├─────────────────────────────────────────────────────────────┤
  │ 第3层：深度技术剖析（Ch8-12）                                │
  │ ┌──────────────┬──────────────────────────────────────────┐ │
  │ │ Ch8 ClaudeCode│ nO主循环/h2A实时Steering/上下文压缩/SubAgent│
  │ │ Ch9 RAG深度   │ Naive→Advanced→GraphRAG→Agentic RAG      │
  │ │ Ch10 MCP协议  │ JSON-RPC/原语/能力协商/stdio/SSE          │
  │ │ Ch11 ToolCall │ OpenAI vs Anthropic/Streaming/Strict     │
  │ │ Ch12 基础设施  │ OpenClaw架构/Harness/Agent生产化Checklist │
  │ └──────────────┴──────────────────────────────────────────┘ │
  ├─────────────────────────────────────────────────────────────┤
  │ 第4层：工程化与前沿（Ch13-18）                                │
  │ ┌──────────────┬──────────────────────────────────────────┐ │
  │ │ Ch13 FastAPI  │ REST/SSE/WebSocket Agent 服务化部署      │
  │ │ Ch14 SQLite   │ 5表Schema + WAL + 会话/任务/用户管理     │
  │ │ Ch15 A2A协议   │ AgentCard/Task/Artifact + 多Agent协作    │
  │ │ Ch16 MemGPT   │ Core Memory/Heartbeat/Sleep-Time/FS记忆  │
  │ │ Ch17 CompUse  │ Screenshot-Action Loop/坐标/安全沙箱     │
  │ │ Ch18 安全防护  │ Prompt Injection/权限分级/审计/4层防御   │
  │ └──────────────┴──────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────┘

学习建议：
  1. 如果你是 Agent 新手：按 Ch1→Ch18 顺序学习
  2. 如果你已有基础：直接跳到 Ch8 开始深度技术
  3. 如果你要准备面试：Ch7(面试题) + Ch8(Claude Code) + Ch10(MCP)
                       + Ch11(ToolCall) + Ch15(A2A) + Ch18(安全)
  4. 如果你要构建产品：Ch13(FastAPI) + Ch14(SQLite) + Ch12(生产)

---

0.5 前置知识自查清单
━━━━━━━━━━━━━━━━━━━

学习本课程前，你需要具备：

  ✅ Python 基础
     - 会写函数、类、装饰器
     - 会用 pip 安装包
     - 理解异步编程（async/await）的概念（Ch1-12 了解即可，Ch13 必须）
     - 了解 SQL 基础（Ch14 需要）

  ✅ LLM 基础认知
     - 用过 ChatGPT / Claude / 文心一言 等产品
     - 知道什么是 Prompt（提示词）
     - 了解 Token 是什么
     - 了解 Function Calling 的概念（Ch1 会详细讲）

  ✅ 基本概念
     - 知道什么是 API 调用 / HTTP 协议
     - 了解 JSON 数据格式
     - 理解「函数」和「函数调用」的区别
     - 了解 Git 基本操作

  ⚠️ 不需要：
     - 不需要深度学习理论基础（不会训练模型）
     - 不需要 C++ / CUDA
     - 不需要分布式系统经验
     - 不需要前端开发能力

---

0.6 环境搭建
━━━━━━━━━━━

下面是一键安装脚本，运行本文件即可检查环境。
如需运行后续章节的 Agent 代码，需要安装如下依赖：

  ┌────────────────────┬────────────────────────────────────┐
  │       章节          │            需要的依赖               │
  ├────────────────────┼────────────────────────────────────┤
  │ Ch1-3 基础理论     │ openai, python-dotenv              │
  │ Ch4-5 框架         │ langchain, langchain-openai,      │
  │                    │ langgraph                         │
  │ Ch8-12 深度技术    │ pydantic, httpx, tiktoken         │
  │ Ch13 FastAPI       │ fastapi, uvicorn                  │
  │ Ch14 SQLite        │ 标准库（无需额外安装）              │
  │ Ch15-18 其他       │ 无需额外依赖                       │
  └────────────────────┴────────────────────────────────────┘
"""

import subprocess
import sys


def check_python_version():
    """检查 Python 版本是否满足要求（>= 3.10）。"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"[✓] Python {version.major}.{version.minor}.{version.micro} - 满足要求")
        return True
    print(f"[✗] Python {version.major}.{version.minor} - 需要 Python 3.10+")
    return False


def install_dependencies():
    """一键安装本课程所需的所有核心依赖。"""
    packages = [
        "openai>=1.0.0",           # OpenAI API 调用 (Ch1-12)
        "langchain>=0.3.0",       # LangChain 核心框架 (Ch4-5)
        "langchain-openai",        # LangChain OpenAI 集成 (Ch4-5)
        "langgraph",               # LangGraph 状态机 (Ch4-5)
        "pydantic>=2.0.0",        # 数据验证 (Ch13)
        "python-dotenv",           # 环境变量管理
        "httpx",                   # HTTP 请求
        "tiktoken",                # Token 计数
        "fastapi",                 # Web 框架 (Ch13)
        "uvicorn",                 # ASGI 服务器 (Ch13)
    ]
    for pkg in packages:
        print(f"正在安装 {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"]
        )
    print("\n[✓] 所有依赖安装完成！")


def show_env_setup_guide():
    """展示 API Key 配置说明。"""
    guide = """
    ╔══════════════════════════════════════════════════════════╗
    ║              API Key 配置指南                             ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  1. 在项目根目录创建 .env 文件：                           ║
    ║     OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx               ║
    ║     OPENAI_BASE_URL=https://api.openai.com/v1            ║
    ║                                                          ║
    ║  2. 如使用国产模型（如 DeepSeek / 通义千问）：              ║
    ║     OPENAI_API_KEY=your-deepseek-key                     ║
    ║     OPENAI_BASE_URL=https://api.deepseek.com/v1          ║
    ║     LLM_MODEL=deepseek-chat                             ║
    ║                                                          ║
    ║  3. 验证配置：                                            ║
    ║     python -c "from openai import OpenAI;                ║
    ║       import os; from dotenv import load_dotenv;         ║
    ║       load_dotenv();                                    ║
    ║       client = OpenAI();                                ║
    ║       print(client.models.list().data[0].id)"            ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(guide)


if __name__ == "__main__":
    print("=" * 60)
    print("   AI Agent 课程 - 环境检查与安装")
    print("=" * 60)
    print()

    if not check_python_version():
        sys.exit(1)

    print()
    print("后续章节运行前，请先执行依赖安装：")
    print("  修改下方 install=True 后运行本文件")
    print()

    install = False  # 改为 True 以执行安装
    if install:
        install_dependencies()

    show_env_setup_guide()
