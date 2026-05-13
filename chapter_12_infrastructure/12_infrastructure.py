"""
第12章：OpenClaw / Harness 与 Agent 生产基础设施
================================================

📌 本章目标：
  1. 深入理解 OpenClaw 的 Gateway 中心化架构
  2. 掌握 AgentSkills 和 ClawHub 的插件生态
  3. 了解 Harness (OpenHarness) 作为开源编码 Agent 的设计亮点
  4. 理解 Agent 生产基础设施的全景图（Tracing / 评测 / 安全 / 部署）
  5. 学会 Agent 评测框架 MultiAgentEval 的使用方式

📌 面试高频点：
  - OpenClaw 和 Claude Code 的架构差异？
  - 什么是 Agent 的 Gateway 架构？有什么优势？
  - Agent 生产化的基础设施包括哪些？
  - 如何搭建 Agent 的评测体系？


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
12.1 OpenClaw —— 145K Star 的现象级开源 Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OpenClaw 的故事：
  2025年11月：奥地利开发者 Peter Steinberger 发布 Clawdbot
  2025年12月：更名为 Moltbot
  2026年初：再次更名为 OpenClaw，GitHub Star 数突破 145K
  背书方：Cloudflare、DigitalOcean、IBM

它「不是一个 Agent 框架」，它「就是 Agent 本身」。
—— 这条定位把它和 LangChain/crewAI 区隔开来。

OpenClaw 的 Gateway 中心化架构：

  ┌─────────────────────────────────────────────────────────┐
  │                    The Gateway                           │
  │                  (Node.js 控制平面)                       │
  │                                                          │
  │  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
  │  │ 消息路由   │  │ 状态管理中心  │  │ 安全策略执行      │  │
  │  │          │  │              │  │                  │  │
  │  └──────────┘  └──────────────┘  └──────────────────┘  │
  │                                                          │
  │  ┌──────────────────────────────────────────────────┐  │
  │  │             LLM Provider Abstraction             │  │
  │  │   Claude | GPT | Gemini | Ollama | 任意兼容API    │  │
  │  └──────────────────────────────────────────────────┘  │
  │                                                          │
  │  ┌──────────────────────────────────────────────────┐  │
  │  │            Persistent Memory Store               │  │
  │  │         长期记忆 + 用户偏好 + 对话历史             │  │
  │  └──────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────┘
           │                  │                  │
    ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐
    │  Channel 1   │   │  Channel 2   │   │  Channel 3   │
    │  WhatsApp    │   │  Telegram    │   │   Discord    │
    └─────────────┘   └─────────────┘   └─────────────┘

Gateway 的职责：
  1. 消息路由：从不同 Channel 接收消息，路由到正确的 LLM Provider
  2. 状态管理：维护每个用户的对话状态和 Agent 运行状态
  3. 安全策略：权限控制、速率限制、内容过滤
  4. Provider 抽象：统一接口，可热切换底层 LLM

Channel Integrations（消息渠道）：
  OpenClaw 不是只能通过 CLI 交互，它可以接入：
  - WhatsApp
  - Telegram
  - Discord
  - Slack
  - iMessage
  - Web Chat

  Agent 在这些渠道上表现得像一个「真人」。


12.2 OpenClaw 的 AgentSkills 和 ClawHub
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是 OpenClaw 最具特色的设计。

AgentSkills：
  - 类似 Node.js 的 npm 包
  - 社区可以开发、发布、安装 Agent 的「技能」
  - 每个 Skill 是一个独立的模块（含工具定义 + 实现）

  示例 Skill：
    skill/weather         → 天气查询
    skill/calendar        → 日历管理
    skill/email           → 邮件发送
    skill/translation     → 翻译服务
    skill/code-review     → 代码审查

ClawHub：
  - AgentSkills 的注册中心
  - 类似 npm registry 或 Docker Hub
  - 社区贡献和共享 Skill

与 MCP 的关系：
  - MCP 定义了「协议」—— 如何连接和通信
  - AgentSkill 定义了「能力」—— 具体做什么
  - 一个 AgentSkill 可以通过 MCP 协议暴露给 LLM
  - 两者是协议层和功能层的关系


12.3 Harness (OpenHarness) —— 开源编码 Agent 黑马
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Harness 是 2026 年崛起的开源编码 Agent。

核心亮点：
  - CLI + SDK 双模式（既可以命令行用，也可以嵌入代码）
  - 支持任意 LLM（Claude、GPT、Gemini、Ollama 等）
  - 在 Harness-Bench 上获得 100%
  - 多提供商兼容（一个命令切换模型）

Harness 的 REPL 命令体系：

  ┌────────────────┬──────────────────────────────────────┐
  │     命令        │               功能                    │
  ├────────────────┼──────────────────────────────────────┤
  │ /help          │ 帮助和提示                            │
  │ /connect       │ 配置 API Key                         │
  │ /model         │ 切换模型（/model gpt-5.2）            │
  │ /plan          │ 用只读 Agent 规划实现方案              │
  │ /review        │ 审查代码变更                          │
  │ /team          │ 分解任务并并行执行多个 Agent           │
  │ /status        │ 显示提供商、模型、会话、成本           │
  │ /cost          │ 显示 Token 使用量和费用               │
  └────────────────┴──────────────────────────────────────┘

Harness 的权限模式：
  - 默认模式：每个工具调用需要用户确认
  - Bypass 模式：完全自动批准（适合 CI/CD 脚本）

应用场景：
  harness "Fix the authentication bug in auth.py"
  harness --permission bypass "Run all tests and fix failures"
  harness -p ollama -m llama3.3 "Write unit tests for utils.py"

与 Claude Code 的对比：
  ┌──────────────┬────────────────────┬───────────────────┐
  │     维度      │      Harness        │    Claude Code     │
  ├──────────────┼────────────────────┼───────────────────┤
  │ 源           │ 开源                │ 闭源（但可逆向）    │
  │ 模型绑定      │ 任何模型            │ Claude 系列        │
  │ 本地模型      │ 支持 (Ollama)       │ 不支持             │
  │ 多Agent      │ /team 命令          │ Task 工具          │
  │ 成熟度        │ Alpha 阶段          │ 生产级别            │
  │ 社区          │ 快速发展中          │ 官方支持            │
  └──────────────┴────────────────────┴───────────────────┘


12.4 Agent 评测基础设施 —— MultiAgentEval
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MultiAgentEval 是一个企业级的 Agent 评测框架。

核心概念：

  1. Scenario（评测场景）
     定义要评测什么：场景描述 + 任务列表 + 成功标准

  2. Task（评测任务）
     每个 Scenario 包含多个 Task：
     - task_id：任务ID
     - description：Agent 收到的提示词
     - success_criteria：评测指标 + 阈值
     - required_tools：Agent 应该调用的工具
     - expected_state_changes：期望的状态变化

  3. Tool Sandbox（工具沙箱）
     - 模拟工具调用（不调用真实 API）
     - 工具行为由 Scenario 定义控制
     - 支持策略检查（Policy Check）

  4. Metrics（评测指标）
     - policy_compliance：策略合规性
     - path_parsimony：执行效率（步骤越少越好）
     - state_verification：状态验证
     - calculation_accuracy：计算准确性
     - planning_quality：规划质量
     - root_cause_analysis_correctness：诊断准确性
     - consistency_score：多次运行一致性
     - luna_judge_score：LLM-as-Judge 语义评测

评测流程：
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 加载场景  │───→│ 执行任务  │───→│ 收集指标  │───→│ 生成报告  │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
                       │
                       ▼
               ┌───────────────┐
               │ Tool Sandbox  │  ← 模拟工具调用
               │ Policy Engine │  ← 检查合规性
               └───────────────┘


12.5 Agent 生产基础设施全景图
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

一个生产级 Agent 系统需要以下基础设施：

┌─────────────────────────────────────────────────────────────┐
│                      生产基础设施全景                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│   可观测性       │      评测        │        安全              │
│  ─────────────  │  ─────────────  │  ─────────────────────  │
│  • LangSmith     │  • MultiAgentEval│  • Prompt Injection防护 │
│  • LangFuse      │  • RAGAS        │  • 权限最小化            │
│  • OpenTelemetry │  • AgentBench   │  • 审计日志              │
│  • 自定义 Tracing│  • SWE-bench    │  • 数据脱敏              │
│  • 日志聚合      │  • 自定义评测集   │  • Rate Limiting       │
├─────────────────┼─────────────────┼─────────────────────────┤
│   成本控制       │      部署        │        可靠性            │
│  ─────────────  │  ─────────────  │  ─────────────────────  │
│  • 模型分层      │  • Docker/K8s    │  • 重试机制              │
│  • 语义缓存      │  • Serverless    │  • 超时控制              │
│  • Token 预算    │  • 负载均衡      │  • 熔断器                │
│  • 批处理        │  • 蓝绿部署      │  • 降级策略              │
│  • 使用配额      │  • Feature Flag  │  • 健康检查              │
└─────────────────┴─────────────────┴─────────────────────────┘

可观测性（最重要！）：
  - 每次 LLM 调用：model + input_tokens + output_tokens + latency
  - 每次 Tool Call：tool_name + args + result + latency
  - 每个 Agent 步骤：step_number + action + observation
  - 指标看板：成功率、平均延迟、Token 消耗趋势、错误分布

  LangSmith / LangFuse 的使用：
    ┌──────────┐     ┌──────────────┐     ┌──────────┐
    │ Agent    │────→│ LangSmith    │────→│ 可视化    │
    │ 执行过程  │     │ (Tracing)    │     │ Dashboard │
    └──────────┘     └──────────────┘     └──────────┘
                            │
                            ▼
                      ┌──────────┐
                      │ 告警系统  │ ← PagerDuty / Slack
                      └──────────┘
"""


class AgentProductionChecklist:
    """Agent 生产化就绪度检查清单。"""

    CHECKLIST = [
        # 可观测性
        ("可观测性", "每次 LLM 调用记录 input/output tokens + latency"),
        ("可观测性", "每次 Tool Call 记录工具名 + 参数 + 结果 + 耗时"),
        ("可观测性", "核心指标看板：成功率 / P50/P99延迟 / Token消耗趋势"),
        ("可观测性", "告警规则：成功率 < 95% → 通知"),
        # 评测
        ("评测", "离线评测集覆盖典型场景 + 边界条件 + Bad Case"),
        ("评测", "上线前回归评测（确保新版本不退化）"),
        ("评测", "A/B 测试框架（对比新老版本效果）"),
        # 安全
        ("安全", "Prompt Injection 检测与过滤"),
        ("安全", "工具调用分级：读自动 / 写确认 / 危险需二次确认"),
        ("安全", "审计日志：记录所有操作，可追溯"),
        ("安全", "敏感信息脱敏（API Key、用户数据等）"),
        # 成本
        ("成本", "模型分层：简单任务用小模型 (gpt-4o-mini)"),
        ("成本", "语义缓存：避免重复查询消耗 Token"),
        ("成本", "Token 预算告警：单用户/单日/单月上限"),
        # 可靠性
        ("可靠性", "LLM 调用重试 (exponential backoff)"),
        ("可靠性", "超时控制 (30s 超时 + 降级策略)"),
        ("可靠性", "熔断器：连续失败 N 次 → 暂停 → 恢复"),
        ("可靠性", "优雅降级：LLM 不可用时返回兜底回答"),
    ]

    @classmethod
    def check(cls) -> list[tuple[str, str, bool]]:
        """返回检查清单及每项的完成状态（这里标注为待检查）。

        Returns:
            (类别, 检查项, 是否完成) 的列表。
        """
        return [(cat, item, False) for cat, item in cls.CHECKLIST]

    @classmethod
    def print_checklist(cls):
        """打印检查清单。"""
        for cat, item in cls.CHECKLIST:
            print(f"  [{cat:8s}] ☐ {item}")


"""
12.6 全课程体系梳理 —— 回顾与展望
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

现在你学完了全部 12 章：

  Ch0   → 课程概览，理解 Agent 是什么
  Ch1-3 → Agent 的基础理论（循环、组件、类型）
  Ch4-5 → Agent 的工程实践（框架、Multi-Agent）
  Ch6-7 → Agent 的评估与求职（评测、面试）
  Ch8   → Claude Code 架构（nO/h2A/Compaction/SubAgent）
  Ch9   → RAG 深度（Naive→Advanced→GraphRAG→Agentic RAG）
  Ch10  → MCP 协议（JSON-RPC/原语/能力协商/传输层）
  Ch11  → Tool Calling 底层（OpenAI vs Anthropic/Streaming/Strict）
  Ch12  → 生产基础设施（OpenClaw/Harness/评测/可观测性）← 你在这里

面试时你可以自信地讨论：
  ✓ Agent 的核心循环和组件
  ✓ Function Calling 的底层原理
  ✓ MCP 协议的架构和调用流程
  ✓ RAG 的技术演进和 GraphRAG
  ✓ Claude Code 的设计哲学
  ✓ Multi-Agent 系统的架构模式
  ✓ Agent 的生产化 Checklist

核心能力矩阵：
┌──────────────────┬──────────────────────────┬─────────┐
│      能力         │         覆盖章节          │  深度    │
├──────────────────┼──────────────────────────┼─────────┤
│ Agent 基础理论    │ Ch1, Ch2, Ch3            │  ⭐⭐⭐  │
│ 框架上手          │ Ch4, Ch5                 │  ⭐⭐⭐  │
│ RAG 技术栈        │ Ch9                      │  ⭐⭐⭐⭐ │
│ MCP 协议          │ Ch10                     │  ⭐⭐⭐⭐ │
│ Tool Calling     │ Ch11                     │  ⭐⭐⭐⭐⭐│
│ Claude Code 架构  │ Ch8                      │  ⭐⭐⭐⭐⭐│
│ 生产基础设施      │ Ch6, Ch12                │  ⭐⭐⭐  │
│ 面试准备          │ Ch7                      │  ⭐⭐⭐⭐ │
└──────────────────┴──────────────────────────┴─────────┘
"""


"""
12.7 本章总结
━━━━━━━━━━━━━━

核心要点回顾：

1. OpenClaw
   - Gateway 中心化架构（Node.js 控制平面）
   - 多 Channel 接入（WhatsApp/Telegram/Discord）
   - AgentSkills + ClawHub 插件生态
   - 定位：「是 Agent，不是框架」

2. Harness (OpenHarness)
   - CLI + SDK 双模式
   - 支持任意 LLM + 本地模型
   - /team 命令实现并行 Multi-Agent
   - Alpha 阶段，但增长速度惊人

3. Agent 评测基础设施
   - MultiAgentEval：企业级评测框架
   - 核心概念：Scenario → Task → Sandbox → Metrics
   - 8 种内置评测指标

4. 生产 Checklist（面试最爱问！）
   - 可观测性（Tracing + Dashboard + Alerting）
   - 评测体系（离线评测 + A/B测试 + 回归评测）
   - 安全防护（Prompt Injection + 权限分级 + 审计）
   - 成本控制（模型分层 + 缓存 + 预算）
   - 可靠性（重试 + 超时 + 熔断 + 降级）

面试速记：
  "Agent 怎么上生产？"
  → 可观测性 + 评测体系 + 安全防护 + 成本控制 + 可靠性
  → 每个环节都可以展开讲具体方案
  → 关键在于「工程化思维」而非「Demo 思维」
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第12章：OpenClaw / Harness 与 Agent 生产基础设施      ║")
    print("║  Gateway架构 · AgentSkills · 评测 · 生产Checklist    ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n▶ 12.1 OpenClaw 架构概览")
    print("-" * 50)
    oc_features = [
        "Gateway 中心化架构（Node.js 控制平面）",
        "多 Channel 接入：WhatsApp / Telegram / Discord / Slack",
        "LLM Provider 抽象：Claude / GPT / Gemini / Ollama",
        "持久化记忆存储：用户偏好 + 对话历史",
        "AgentSkills + ClawHub：Agent 技能生态",
        "145K+ GitHub Stars",
    ]
    for f in oc_features:
        print(f"  • {f}")

    print("\n▶ 12.2 AgentSkills vs MCP 关系")
    print("-" * 50)
    print("  MCP 定义「协议」—— 如何连接和通信")
    print("  AgentSkill 定义「能力」—— 具体做什么")
    print("  关系：AgentSkill 通过 MCP 协议暴露给 LLM")
    print("  类比：MCP = USB-C标准，AgentSkill = USB设备")

    print("\n▶ 12.3 Harness 亮点")
    print("-" * 50)
    h_features = [
        "CLI + SDK 双模式",
        "支持任意 LLM（Claude/GPT/Gemini/Ollama）",
        "Harness-Bench 100% 评分",
        "/plan /review /team /status /cost 命令",
        "Bypass 模式（CI/CD 友好）",
    ]
    for f in h_features:
        print(f"  • {f}")

    print("\n▶ 12.5 Agent 生产化 Checklist")
    print("-" * 50)
    AgentProductionChecklist.print_checklist()

    print("\n▶ 12.6 全课程回顾")
    print("-" * 50)
    chapters = [
        "Ch0   → 课程概览",
        "Ch1-3 → Agent 基础理论（循环/组件/类型）",
        "Ch4-5 → 工程实践（框架/Multi-Agent）",
        "Ch6-7 → 评估与求职（评测/面试）",
        "Ch8   → Claude Code 架构深度剖析",
        "Ch9   → RAG 技术全览（Naive→Agentic RAG）",
        "Ch10  → MCP 协议详解",
        "Ch11  → Tool Calling 底层机制",
        "Ch12  → 生产基础设施（OpenClaw/Harness/评测）",
    ]
    for ch in chapters:
        print(f"  {ch}")

    print("\n" + "=" * 60)
    print("  🎓 全课程体系完成！")
    print("  从 Agent 基础到 Claude Code 逆向分析")
    print("  从 RAG 到 MCP 到 Tool Calling 底层原理")
    print("  你现在已经具备了成为顶级 Agent 工程师的所有知识")
    print("=" * 60)
