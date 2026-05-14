<p align="center">
  <h1 align="center">🤖 AI Agent 全栈学习课程</h1>
  <p align="center">
    从零到一，系统掌握 AI Agent 核心理论与工程实践<br>
    28 章节 · 18000+ 行代码 · 50+ 可运行示例 · 面试全覆盖
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/章节-28-green" alt="Chapters">
  <img src="https://img.shields.io/badge/许可-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/更新-2026.05-brightgreen" alt="Update">
</p>

---

## 📖 项目简介

这是一套**面向求职**的 AI Agent 全栈学习课程，从 Agent 基础理论到 Claude Code 逆向工程、从 RAG 到 MCP/A2A 协议、从 DSPy 到生产可观测性，覆盖 **28 个主题、6 个层次**。每个章节都是 **可独立运行的 `.py` 文件**，既是完整讲义，又是可执行代码。

> **适合人群**：应届毕业生、转行工程师、任何想系统学习 AI Agent 的开发者。

---

## 🗺️ 课程路线图（28 章 · 6 层递进）

```
第1层：理论基础 ── 第2层：工程实践 ── 第3层：深度技术 ── 第4层：工程化与前沿
   Ch1-3               Ch4-7              Ch8-12              Ch13-18

第5层：高级架构 ── 第6层：基础补强
  Ch19-24            Ch25-28
```

### 第1层：Agent 理论基础

| 章节 | 内容 | 关键技术 |
|------|------|----------|
| **Ch0** | 课程概览与环境搭建 | 学习路线图、依赖安装、API Key 配置 |
| **Ch1** | 第一个 Agent | 裸写 ReAct 循环、Function Calling 原理 |
| **Ch2** | Agent 核心组件 | 规划器、记忆系统（短期/长期/工作）、工具设计黄金法则 |
| **Ch3** | Agent 类型分类 | ReAct / Plan-Execute / Reflexion 对比 |

### 第2层：工程实践与框架

| 章节 | 内容 | 关键技术 |
|------|------|----------|
| **Ch4** | 主流框架实战 | LangChain Agent + LangGraph 状态机 |
| **Ch5** | 多智能体系统 | Writer+Reviewer 协作、crewAI 风格 |
| **Ch6** | 评估与测试 | 评测框架、LLM-as-Judge、生产 Checklist |
| **Ch7** | 求职面试准备 | 20 道高频面试题 + 项目指南 + 面试流程 |

### 第3层：深度技术剖析

| 章节 | 内容 | 关键技术 |
|------|------|----------|
| **Ch8** | Claude Code 架构 | nO 主循环、h2A Steering、上下文压缩、SubAgent |
| **Ch9** | RAG 深度剖析 | Naive→Advanced→GraphRAG→Agentic RAG + RAGAS 评估 |
| **Ch10** | MCP 协议详解 | JSON-RPC、原语（Tools/Resources/Prompts）、能力协商 |
| **Ch11** | Tool Calling 底层 | OpenAI vs Anthropic、Streaming 组装、Strict 模式 |
| **Ch12** | Agent 生产基础设施 | OpenClaw 架构、Harness、MultiAgentEval、生产化 Checklist |

### 第4层：工程化与前沿

| 章节 | 内容 | 关键技术 |
|------|------|----------|
| **Ch13** | FastAPI 服务化 | REST API、SSE 流式、WebSocket、生产部署架构 |
| **Ch14** | SQLite 持久化 | 5 表 Schema、WAL 模式、会话/任务/用户管理 |
| **Ch15** | Google A2A 协议 | AgentCard、Task、Artifact、Multi-Agent 协作 |
| **Ch16** | MemGPT/Letta 记忆 | Core Memory、Heartbeat、Sleep-Time、Filesystem Memory |
| **Ch17** | Computer Use | Screenshot-Action Loop、坐标计算、安全沙箱 |
| **Ch18** | Agent 安全与护栏 | Prompt Injection 攻防、权限分级、输入消毒、审计 |

### 第5层：高级架构与优化

| 章节 | 内容 | 关键技术 |
|------|------|----------|
| **Ch19** | Agentic Workflow 设计模式 | Reflection / Routing / Orchestrator-Worker / Evaluator-Optimizer 等 7 种模式 |
| **Ch20** | Context Engineering | Context Rot 原理、预算管理、XML 结构化 Prompt、Skill.md |
| **Ch21** | Streaming & 实时架构 | EventBus、动态中断、背压控制、StateManager Reducer |
| **Ch22** | DSPy 自动优化 | Signature→Module→Optimizer、自动 few-shot、与 LangChain 互补 |
| **Ch23** | 代码 Agent 架构横评 | CodeAct / ACI / Plan-Execute、SWE-bench、Agentless 发现 |
| **Ch24** | Agent 可观测性 | Tracing Span 树、Dashboard、LangSmith vs LangFuse、告警规则 |

### 第6层：基础能力补强

| 章节 | 内容 | 关键技术 |
|------|------|----------|
| **Ch25** | 向量数据库选型 | Chroma/Pinecone/Milvus/Qdrant 对比、Embedding 维度权衡 |
| **Ch26** | 模型路由策略 | Threshold / Cascade / Semantic / Cost-Aware 四种路由 |
| **Ch27** | Agent Prompt 工程 | System Prompt 6 模块模板、工具描述评分卡 |
| **Ch28** | 语义缓存与 Token 优化 | Exact→Semantic→LLM 三级缓存、Token 预算管理 |

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Callous-0923/agent-study.git
cd agent-study
```

### 2. 环境检查

```bash
python chapter_00_overview/00_course_overview.py
```

### 3. 安装依赖

打开 `chapter_00_overview/00_course_overview.py`，将 `install = False` 改为 `install = True`，然后运行：

```bash
python chapter_00_overview/00_course_overview.py
```

### 4. 配置 API Key（仅 Ch1-5 需要）

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

> 也可使用国产模型（DeepSeek / 通义千问），修改 `OPENAI_BASE_URL` 和 `LLM_MODEL` 即可。

### 5. 开始学习（大部分章节无需 API Key！）

```bash
# 无需 API Key，直接运行
python chapter_08_claude_code/08_claude_code_architecture.py
python chapter_10_mcp/10_mcp_deepdive.py
python chapter_14_sqlite/14_sqlite_agent_storage.py
python chapter_19_workflow_patterns/19_workflow_patterns.py
python chapter_21_streaming/21_streaming_architecture.py
python chapter_24_observability/24_observability.py
python chapter_26_model_routing/26_model_routing.py
python chapter_28_cache/28_cache.py
```

---

## 📦 依赖说明

| 依赖 | 使用章节 | 说明 |
|------|----------|------|
| `openai` | Ch1-3 | OpenAI API 调用 |
| `langchain` + `langgraph` | Ch4-5 | Agent 框架实战 |
| `fastapi` + `uvicorn` | Ch13 | Agent 服务化部署 |
| `pydantic` | Ch13 | 数据模型校验 |
| `python-dotenv` | Ch0 | 环境变量管理 |

> **注**：Ch8-12、Ch14-28 绝大部分章节仅依赖 Python 标准库（`sqlite3`、`asyncio`、`hashlib` 等），无需额外安装即可运行。

---

## 🎯 学习建议

### 路径 1：从零开始（推荐新手）
按 Ch1 → Ch28 顺序学习，每章 1-2 小时。

### 路径 2：面试突击（重点章节）
- **Ch7**：20 道高频面试题 + 面试流程
- **Ch8**：Claude Code 架构（工业级 Agent 设计）
- **Ch10 + Ch15**：MCP & A2A 双协议
- **Ch11**：Tool Calling 底层机制
- **Ch18**：Agent 安全（区分 Demo vs 生产工程师）
- **Ch19**：Workflow 设计模式（系统设计万能框架）
- **Ch26**：模型路由（降本 50-80%）

### 路径 3：构建产品
- **Ch13**：FastAPI 服务化
- **Ch14**：SQLite 持久化
- **Ch12 + Ch24**：生产化 Checklist + 可观测性
- **Ch26 + Ch28**：成本优化（路由 + 缓存）

---

## 🧠 核心技术覆盖

```
Tool Calling 底层     ★★★★★  OpenAI/Anthropic 两套实现完整对比 + Streaming 组装
MCP 协议              ★★★★★  完整生命周期模拟（Initialize→tools/call）
A2A 协议              ★★★★★  AgentCard/Task/Artifact + Multi-Agent 协作
Claude Code 架构      ★★★★★  nO/h2A/Compaction/SubAgent 逆向分析
RAG 全栈              ★★★★   Naive→Advanced→GraphRAG→Agentic RAG
模型路由              ★★★★   4 种策略 + 成本对比实验（节省 94%）
语义缓存              ★★★★   三级缓存 + Token 预算管理
Agent 安全            ★★★★   Prompt Injection + 权限分级 + 4 层防御
DSPy 自动优化         ★★★★   Signature/Module/Optimizer + LangChain 互补
Agentic Workflow      ★★★★   7 种设计模式 + 系统设计答题框架
Context Engineering   ★★★★   Context Rot 原理 + XML Prompt + 预算管理
Streaming 实时架构    ★★★★   EventBus + 动态中断 + 背压控制
可观测性              ★★★★   Tracing Span 树 + LangSmith vs LangFuse
MemGPT 记忆           ★★★★   Core Memory/Heartbeat/Sleep-Time/Filesystem
代码 Agent            ★★★    CodeAct/ACI/Plan-Execute + SWE-bench
向量数据库            ★★★    Chroma/Pinecone/Milvus/Qdrant + Embedding 策略
FastAPI 服务化        ★★★    REST/SSE/WebSocket + 生产部署架构
SQLite 持久化         ★★★    5 表 Schema + WAL + 审计查询
Computer Use          ★★★    Screenshot-Action Loop + 安全沙箱
```

---

## 📁 项目结构

```
agent-study/
├── README.md
├── .gitignore
├── chapter_00_overview/              🚀 课程概览 + 环境搭建
├── chapter_01_fundamentals/          📖 第一个 Agent（裸写 ReAct）
├── chapter_02_components/            🧩 规划器 + 记忆 + 工具设计
├── chapter_03_types/                 🎯 ReAct / Plan-Execute / Reflexion
├── chapter_04_frameworks/            🔧 LangChain + LangGraph
├── chapter_05_multi_agent/           🤝 多智能体协作
├── chapter_06_evaluation/            📊 评测 + 测试策略
├── chapter_07_interview/             🎓 20道面试题 + 求职指南
├── chapter_08_claude_code/           🏗️ Claude Code 架构逆向
├── chapter_09_rag_deepdive/          🔍 RAG 全栈（Naive→Agentic RAG）
├── chapter_10_mcp/                   🔌 MCP 协议完整实现
├── chapter_11_tool_calling/          ⚙️ Tool Calling 底层原理
├── chapter_12_infrastructure/        🏭 OpenClaw/Harness/生产基础设施
├── chapter_13_fastapi/               🌐 FastAPI Agent 服务化
├── chapter_14_sqlite/                💾 SQLite 持久化存储
├── chapter_15_a2a/                   🤖 Google A2A 协议
├── chapter_16_memgpt/                🧠 MemGPT/Letta 记忆架构
├── chapter_17_computer_use/          🖥️ Computer Use + GUI
├── chapter_18_security/              🛡️ Agent 安全与护栏
├── chapter_19_workflow_patterns/     🏷️ Agentic Workflow 设计模式
├── chapter_20_context_engineering/   📐 Context Engineering
├── chapter_21_streaming/             📡 EventBus 实时架构
├── chapter_22_dspy/                  🔬 DSPy 自动优化
├── chapter_23_code_agents/           📊 代码 Agent 架构横评
├── chapter_24_observability/         📈 可观测性（LangSmith/LangFuse）
├── chapter_25_vectordb/              🗄️ 向量数据库选型
├── chapter_26_model_routing/         🔀 模型路由与成本优化
├── chapter_27_prompt_eng/            ✍️ Agent Prompt 工程
└── chapter_28_cache/                 ⚡ 语义缓存与 Token 优化
```

---

## ✨ 特点

- **📝 讲义即代码**：每个 `.py` 文件既是完整讲义（模块级 docstring），又是可运行代码
- **🤖 无需 API Key**：Ch8-28 绝大部分章节仅依赖标准库，可直接运行
- **🎤 面试导向**：每章标注面试高频考点 + 回答框架 + 得分点
- **🔗 前后关联**：章节间通过引用形成完整知识网络
- **📊 可运行演示**：每个章节都包含完整的演示输出
- **🇨🇳 中文优先**：全中文讲义 + 代码注释

---

## 🌟 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., ICLR 2023)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (Packer et al., 2023)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023)
- [DSPy: Compiling Declarative LM Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714) (Khattab et al., 2023)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) (Jimenez et al., 2024)
- [Agentless: Demystifying LLM-Based Software Engineering Agents](https://arxiv.org/abs/2407.01489) (Xia et al., 2024)
- [MCP 官方规范](https://modelcontextprotocol.io) (Anthropic, 2024-2025)
- [A2A 协议规范](https://a2a-protocol.org) (Google, 2025)
- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic, 2024)
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) (Anthropic, 2025)
- [Claude Code 逆向分析](https://github.com/shareAI-lab/analysis_claude_code) (Community, 2025)
- [OpenClaw](https://github.com/openclaw/openclaw) (Peter Steinberger, 2025)
- [Harness (OpenHarness)](https://github.com/AgentBoardTT/openharness) (2026)

---

## 📄 许可

MIT License — 自由使用、修改、分发。

---

<p align="center">
  <b>如果这个项目对你有帮助，请给一个 ⭐ Star！</b><br>
  <sub>28 章 · 6 层递进 · 持续更新中 · 欢迎提交 Issue 和 PR</sub>
</p>
