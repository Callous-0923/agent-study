"""
第0章：AI Agent 课程概览与前置知识
===================================

📌 本章目标：
  1. 理解「什么是 AI Agent」- 建立直觉认知
  2. 掌握 Agent 的核心组成公式：Agent = LLM + 规划 + 记忆 + 工具
  3. 了解 Agent 开发的完整技术栈和 36 章学习路线
  4. 确认学习本课程所需的前置知识

📌 本章结构：
  0.1 Agent 是什么？（5分钟理解）
  0.2 为什么 2024-2026 年是 Agent 爆发期？
  0.3 Agent 核心公式：Agent = LLM + 规划 + 记忆 + 工具
  0.4 全套学习路线图（36章七层递进）
  0.5 前置知识自查清单
  0.6 环境搭建（一键运行）

---

0.1 Agent 是什么？—— 用类比建立直觉
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

要理解 Agent，先看一个最简单的对比：

❌ 普通 LLM（如 ChatGPT 基础版）：
   用户：「北京明天天气怎么样？」
   LLM：「抱歉，我无法获取实时数据。」  ← 只会"说"，不会"做"

  为什么？因为 LLM 的「世界」只有训练数据。它就像一个被困在房间里
  的人，只能靠记忆回答。它不知道「现在」发生了什么。

✅ AI Agent：
   用户：「北京明天天气怎么样？」
   Agent 内部执行了 4 步：
     1. 思考(Think)：用户想知道天气 → 我需要调用天气查询工具
     2. 行动(Act)：调用 get_weather("北京") → 发送 HTTP 请求到天气 API
     3. 观察(Observe)：返回 {"明天": "晴，25°C"} → 拿到真实数据
     4. 回答：用自然语言组织结果 → 「北京明天晴天，气温 25°C」

核心区别：
  - LLM = 「大脑」（会思考，但只能靠记忆）
  - Agent = 「大脑 + 手」（会思考 + 能获取新信息、执行操作）

这就引出了 Agent 开发的核心思维：我们不是在教 LLM 更多知识，
而是在给它装「手」——让它能查、能算、能操作。这才是 Agent 和
聊天机器人的本质分界线。

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
    • 2025.12: OpenAI 发布 CUA (Computer Using Agent)

  ── 2026: Agent 基础设施成熟 ──
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
  │                  AI Agent                   │
  │  ┌─────────┐  ┌────────┐  ┌──────────────┐  │
  │  │   LLM   │  │ 规划器  │  │   记忆系统    │  │
  │  │ (大脑)   │  │Planning│  │   Memory     │   │ 
  │  └────┬────┘  └───┬────┘  └──────┬───────┘  │
  │       │           │              │          │
  │       └───────────┼──────────────┘          │
  │                   │                         │
  │            ┌──────┴──────┐                  │
  │            │   工具调用    │                 │
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


0.4 全套学习路线图（36章七层递进）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本课程按「理论 → 实践 → 深度 → 工程 → 架构 → 补强 → 专家」七层递进，
共 36 章（Ch0-Ch36），每章既是完整讲义也是可运行代码。

  ┌─────────────────────────────────────────────────────────────┐
  │ 第1层：Agent 理论基础（Ch0-3）                                │
  │ ┌──────────────┬──────────────────────────────────────────┐ │
  │ │ Ch0 课程概览  │ 学习路线图、环境搭建、API Key 配置        │ │
  │ │ Ch1 第一个Agent│ 裸写 ReAct 循环、Function Calling 原理   │ │
  │ │ Ch2 核心组件  │ 规划器 + 记忆系统 + 工具设计黄金法则       │ │
  │ │ Ch3 类型分类  │ ReAct / Plan-Execute / Reflexion 对比    │ │
  │ └──────────────┴──────────────────────────────────────────┘ │
  ├─────────────────────────────────────────────────────────────┤
  │ 第2层：工程实践与框架（Ch4-7）                                │
  │ ┌──────────────┬──────────────────────────────────────────┐ │
  │ │ Ch4 主流框架  │ LangChain Agent + LangGraph 状态机实战    │ │
  │ │ Ch5 多智能体  │ Multi-Agent 协作、Writer+Reviewer 模式    │ │
  │ │ Ch6 评估测试  │ 评测框架 + LLM-as-Judge + 生产 Checklist   │ │
  │ │ Ch7 求职面试  │ 20道高频面试题 + 项目指南 + 面试流程       │ │
  │ └──────────────┴──────────────────────────────────────────┘ │
  ├─────────────────────────────────────────────────────────────┤
  │ 第3层：深度技术剖析（Ch8-12）                                │
  │ ┌──────────────┬──────────────────────────────────────────┐ │
  │ │ Ch8 ClaudeCode│ nO主循环·h2A实时Steering·上下文压缩·SubAgent│
  │ │ Ch9 RAG深度   │ 从Naive到生产级·Chunk·Embedding·RRF     │
  │ │ Ch10 MCP协议  │ JSON-RPC·原语·能力协商·stdio/SSE传输层   │
  │ │ Ch11 ToolCall │ OpenAI vs Anthropic·Streaming·Strict模式 │
  │ │ Ch12 基础设施  │ OpenClaw架构·Harness·Agent生产化Checklist │
  │ └──────────────┴──────────────────────────────────────────┘ │
  ├─────────────────────────────────────────────────────────────┤
  │ 第4层：工程化与前沿（Ch13-18）                                │
  │ ┌──────────────┬──────────────────────────────────────────┐ │
  │ │ Ch13 FastAPI  │ REST API·SSE·WebSocket·生产部署架构      │
  │ │ Ch14 SQLite   │ 5表Schema·WAL模式·会话/任务/用户管理     │
  │ │ Ch15 A2A协议   │ AgentCard·Task·Artifact·多Agent协作      │
  │ │ Ch16 MemGPT   │ Core Memory·Heartbeat·Sleep-Time·FS记忆  │
  │ │ Ch17 CompUse  │ Screenshot-Action Loop·坐标计算·安全沙箱 │
  │ │ Ch18 安全防护  │ Prompt Injection攻防·权限分级·4层防御    │
  │ └──────────────┴──────────────────────────────────────────┘ │
  ├─────────────────────────────────────────────────────────────┤
  │ 第5层：高级架构与优化（Ch19-24）                              │
  │ ┌──────────────┬──────────────────────────────────────────┐ │
  │ │ Ch19 Workflow │ Reflection·Routing·Orchestrator等7种模式 │
  │ │ Ch20 Context  │ Context Rot·预算管理·XML结构化Prompt     │
  │ │ Ch21 Streaming│ EventBus·动态中断·背压控制               │
  │ │ Ch22 DSPy    │ Signature→Module→Optimizer 自动优化       │
  │ │ Ch23 CodeAgent│ CodeAct·ACI·Plan-Execute·SWE-bench横评   │
  │ │ Ch24 可观测   │ Tracing Span树·LangSmith vs LangFuse     │
  │ └──────────────┴──────────────────────────────────────────┘ │
  ├─────────────────────────────────────────────────────────────┤
  │ 第6层：基础能力补强（Ch25-28）                                │
  │ ┌──────────────┬──────────────────────────────────────────┐ │
  │ │ Ch25 向量库   │ Chroma·Pinecone·Milvus·Qdrant对比·Embedding│
  │ │ Ch26 模型路由  │ Threshold·Cascade·Semantic·Cost-Aware 4种 │
  │ │ Ch27 Prompt   │ System Prompt 6模块模板·工具描述评分卡   │
  │ │ Ch28 语义缓存  │ 三级缓存(Exact→Semantic→LLM)·Token预算   │
  │ └──────────────┴──────────────────────────────────────────┘ │
  ├─────────────────────────────────────────────────────────────┤
  │ 第7层：专家级进阶（Ch29-36）                                  │
  │ ┌──────────────┬──────────────────────────────────────────┐ │
  │ │ Ch29 多模态   │ 视觉+文本联合推理·多模态Tool Calling     │
  │ │ Ch30 可靠性   │ 熔断器·指数退避重试·幂等性·降级策略     │
  │ │ Ch31 评测体系  │ GAIA·AgentBench·WebArena·tau-bench      │
  │ │ Ch32 自改进   │ Bad Case收集→自动改Prompt→评测验证      │
  │ │ Ch33 Cache   │ Anthropic Cache·推测解码·KV共享          │
  │ │ Ch34 微调     │ LoRA微调·数据准备·成本收益对比           │
  │ │ Ch35 数据飞轮  │ 交互采集→Bad Case识别→自动触发改进     │
  │ │ Ch36 纵深安全  │ Canary Token·分层隔离·行为沙箱          │
  │ └──────────────┴──────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────┘

学习建议：
  1. 新手入门：按 Ch0 → Ch36 顺序学习，每章 1-2 小时
  2. 有基础者：直接跳到 Ch8 开始深度技术
  3. 面试突击（重点章节）：
     Ch7(面试20问) + Ch8(Claude Code) + Ch9(RAG) + Ch10(MCP)
     + Ch11(ToolCall) + Ch15(A2A) + Ch18(安全) + Ch19(Workflow)
  4. 构建产品：Ch13(FastAPI) + Ch14(SQLite) + Ch24(可观测)
     + Ch26(模型路由) + Ch28(语义缓存)
  5. 降本增效：Ch26(路由节省94%) + Ch28(缓存) + Ch33(Prompt Cache)

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
  │ Ch0 环境检查       │ 标准库（无需额外安装）              │
  │ Ch1-3 基础理论     │ openai, python-dotenv              │
  │ Ch4-5 框架         │ langchain, langchain-openai,      │
  │                    │ langgraph                         │
  │ Ch6-7 评测+面试    │ 无需额外依赖                       │
  │ Ch8-12 深度技术    │ pydantic, httpx, tiktoken         │
  │ Ch13 FastAPI       │ fastapi, uvicorn                  │
  │ Ch14 SQLite        │ 标准库（无需额外安装）              │
  │ Ch15-18 协议/安全  │ 无需额外依赖（标准库）              │
  │ Ch19-21 架构流式   │ 标准库（无需额外安装）              │
  │ Ch22-24 优化可观测  │ 标准库（无需额外安装）              │
  │ Ch25 向量数据库    │ numpy（演示用）                    │
  │ Ch26-28 路由/缓存  │ 标准库（无需额外安装）              │
  │ Ch29-36 专家进阶   │ 标准库（无需额外安装）              │
  └────────────────────┴────────────────────────────────────┘

  ⚡ 核心依赖一行安装（Ch1-18 必需）：
    pip install openai python-dotenv langchain langchain-openai \
                langgraph pydantic httpx tiktoken fastapi uvicorn numpy

  💡 大部分章节（Ch8-36）仅使用 Python 标准库
     （sqlite3 / asyncio / hashlib / json / time），无需额外安装即可运行。
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
        "numpy",                   # 数值计算 (Ch25 向量数据库演示)
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
