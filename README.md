<p align="center">
  <h1 align="center">🤖 AI Agent 全栈学习课程</h1>
  <p align="center">
    从零到一，系统掌握 AI Agent 核心理论与工程实践<br>
    18 章节 · 12000+ 行代码 · 40+ 可运行示例 · 面试全覆盖
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/章节-18-green" alt="Chapters">
  <img src="https://img.shields.io/badge/许可-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/更新-2026.05-brightgreen" alt="Update">
</p>

---

## 📖 项目简介

这是一套**面向求职**的 AI Agent 全栈学习课程，从 Agent 基础理论到 Claude Code 逆向工程，涵盖 **RAG、MCP、A2A、Tool Calling、Computer Use、Agent 安全** 等最新前沿技术。每个章节都是 **可独立运行的 `.py` 文件**，既是完整讲义，又是可执行代码。

> **适合人群**：应届毕业生、转行工程师、任何想系统学习 AI Agent 的开发者。

---

## 🗺️ 课程路线图

```
第1层：理论基础 ──── 第2层：工程实践 ──── 第3层：深度技术 ──── 第4层：工程化与前沿
   Ch1-3               Ch4-7              Ch8-12              Ch13-18
```

| 章节 | 内容 | 关键技术 |
|------|------|----------|
| **Ch0** | 课程概览与环境搭建 | 学习路线图、依赖安装、API Key 配置 |
| **Ch1** | 第一个 Agent | 裸写 ReAct 循环、Function Calling 原理 |
| **Ch2** | Agent 核心组件 | 规划器、记忆系统（短期/长期/工作）、工具设计 |
| **Ch3** | Agent 类型分类 | ReAct / Plan-Execute / Reflexion 对比 |
| **Ch4** | 主流框架实战 | LangChain Agent + LangGraph 状态机 |
| **Ch5** | 多智能体系统 | Writer+Reviewer 协作、crewAI 风格 |
| **Ch6** | 评估与测试 | 评测框架、LLM-as-Judge、生产 Checklist |
| **Ch7** | 求职面试准备 | 20 道高频面试题 + 项目指南 |
| **Ch8** | Claude Code 架构 | nO 主循环、h2A Steering、上下文压缩、SubAgent |
| **Ch9** | RAG 深度剖析 | Naive→Advanced→GraphRAG→Agentic RAG |
| **Ch10** | MCP 协议详解 | JSON-RPC、原语、能力协商、Client-Host-Server |
| **Ch11** | Tool Calling 底层 | OpenAI vs Anthropic、Streaming、Strict 模式 |
| **Ch12** | Agent 生产基础设施 | OpenClaw 架构、Harness、生产化 Checklist |
| **Ch13** | FastAPI 服务化 | REST API、SSE 流式、WebSocket、生产部署 |
| **Ch14** | SQLite 持久化 | 5 表 Schema、WAL 模式、会话/任务/用户管理 |
| **Ch15** | Google A2A 协议 | AgentCard、Task、Artifact、Multi-Agent 协作 |
| **Ch16** | MemGPT/Letta 记忆 | Core Memory、Heartbeat、Sleep-Time Compute |
| **Ch17** | Computer Use | Screenshot-Action Loop、坐标计算、安全沙箱 |
| **Ch18** | Agent 安全与护栏 | Prompt Injection、权限分级、输入消毒、审计 |

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/agent-study.git
cd agent-study
```

### 2. 环境检查

```bash
python chapter_00_overview/00_course_overview.py
```

### 3. 安装依赖

打开 `chapter_00_overview/00_course_overview.py`，将第 258 行的 `install = False` 改为 `install = True`，然后运行：

```bash
python chapter_00_overview/00_course_overview.py
```

### 4. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

> 也可使用国产模型（DeepSeek / 通义千问），只需修改 `OPENAI_BASE_URL` 和 `LLM_MODEL` 即可。

### 5. 开始学习

```bash
# 大部分章节无需 API Key 即可运行
python chapter_08_claude_code/08_claude_code_architecture.py
python chapter_09_rag_deepdive/09_rag_deepdive.py
python chapter_10_mcp/10_mcp_deepdive.py
python chapter_14_sqlite/14_sqlite_agent_storage.py
python chapter_18_security/18_agent_security.py
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

> **注**：Ch8-12、Ch14-18 大部分章节仅依赖 Python 标准库，无需额外安装即可运行。

---

## 🎯 学习建议

### 路径 1：从零开始（推荐新手）
按 Ch1 → Ch18 顺序学习，每章 1-2 小时。

### 路径 2：面试突击
重点学习以下章节：
- **Ch7**：20 道高频面试题
- **Ch8**：Claude Code 架构（面试常问的工业级 Agent 设计）
- **Ch10**：MCP 协议原理
- **Ch11**：Tool Calling 底层机制
- **Ch15**：A2A 协议（2025 最热门话题）
- **Ch18**：Agent 安全（区分 Demo 工程师 vs 生产工程师）

### 路径 3：构建产品
聚焦工程化章节：
- **Ch13**：FastAPI 服务化
- **Ch14**：SQLite 持久化
- **Ch12**：生产化 Checklist

---

## 🧠 核心技术覆盖

```
Tool Calling 底层    ★★★★★  OpenAI/Anthropic 两套实现完整对比
MCP 协议             ★★★★★  完整生命周期模拟（Initialize→tools/call）
Claude Code 架构     ★★★★★  nO/h2A/Compaction/SubAgent 逆向分析
A2A 协议             ★★★★★  AgentCard/Task/Artifact 多Agent协作
RAG 全栈             ★★★★   Naive→Advanced→GraphRAG→Agentic RAG
Agent 安全           ★★★★   Prompt Injection + 权限分级 + 4层防御
MemGPT 记忆          ★★★★   Core Memory/Heartbeat/Sleep-Time
SQLite 持久化        ★★★★   5表Schema + WAL + FTS5 + 审计查询
FastAPI 服务化       ★★★★   REST/SSE/WebSocket + 生产部署架构
Computer Use         ★★★    Screenshot-Action Loop + 坐标计算
```

---

## 📁 项目结构

```
agent-study/
├── README.md                           # 项目说明
├── .gitignore
├── chapter_00_overview/                # 课程概览 + 环境搭建
│   └── 00_course_overview.py
├── chapter_01_fundamentals/            # 第一个 Agent（裸写 ReAct）
│   └── 01_hello_agent.py
├── chapter_02_components/              # Agent 核心组件
│   └── 02_agent_components.py
├── chapter_03_types/                   # Agent 类型分类
│   └── 03_agent_types.py
├── chapter_04_frameworks/              # 主流框架实战
│   └── 04_frameworks.py
├── chapter_05_multi_agent/             # 多智能体系统
│   └── 05_multi_agent.py
├── chapter_06_evaluation/              # 评估与测试
│   └── 06_evaluation.py
├── chapter_07_interview/               # 求职面试准备
│   └── 07_interview_prep.py
├── chapter_08_claude_code/             # Claude Code 架构剖析
│   └── 08_claude_code_architecture.py
├── chapter_09_rag_deepdive/            # RAG 技术全栈
│   └── 09_rag_deepdive.py
├── chapter_10_mcp/                     # MCP 协议详解
│   └── 10_mcp_deepdive.py
├── chapter_11_tool_calling/            # Tool Calling 底层
│   └── 11_tool_calling_deepdive.py
├── chapter_12_infrastructure/          # 生产基础设施
│   └── 12_infrastructure.py
├── chapter_13_fastapi/                 # FastAPI Agent 服务
│   └── 13_fastapi_agent_service.py
├── chapter_14_sqlite/                  # SQLite 持久化存储
│   └── 14_sqlite_agent_storage.py
├── chapter_15_a2a/                     # Google A2A 协议
│   └── 15_a2a_protocol.py
├── chapter_16_memgpt/                  # MemGPT/Letta 记忆
│   └── 16_memgpt_letta.py
├── chapter_17_computer_use/            # Computer Use + GUI
│   └── 17_computer_use.py
└── chapter_18_security/                # Agent 安全与护栏
    └── 18_agent_security.py
```

---

## ✨ 特点

- **📝 讲义即代码**：每个 `.py` 文件既是完整讲义，又是可运行代码
- **🤖 无需 API Key**：Ch8-18 大部分章节仅依赖标准库，可独立运行
- **🎤 面试导向**：每章标注面试高频考点和回答框架
- **🔗 前后关联**：章节间通过交叉引用形成完整的知识网络
- **📊 可运行演示**：每个章节都包含完整的演示输出
- **🇨🇳 中文优先**：全中文讲义 + 代码注释

---

## 🌟 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., ICLR 2023)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (Packer et al., 2023)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023)
- [MCP 官方规范](https://modelcontextprotocol.io) (Anthropic, 2024-2025)
- [A2A 协议规范](https://a2a-protocol.org) (Google, 2025)
- [Claude Code 逆向分析](https://github.com/shareAI-lab/analysis_claude_code) (Community, 2025)
- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic, 2024)
- [OpenClaw](https://github.com/openclaw/openclaw) (Peter Steinberger, 2025)
- [Harness (OpenHarness)](https://github.com/AgentBoardTT/openharness) (2026)

---

## 📄 许可

MIT License — 自由使用、修改、分发。

---

<p align="center">
  <b>如果这个项目对你有帮助，请给一个 ⭐ Star！</b><br>
  <sub>持续更新中 · 欢迎提交 Issue 和 PR</sub>
</p>
