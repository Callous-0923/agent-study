"""
第7章：Agent 求职面试准备与行业实践
===================================

📌 本章目标：
  1. 了解 Agent 相关岗位的类型和要求
  2. 掌握 Agent 面试的高频问题和答题框架
  3. 学会准备有竞争力的项目作品集
  4. 了解行业趋势和发展方向

📌 注意：
  本章不包含代码，是一份完整的面试备战手册。


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7.1 Agent 岗位全景图
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当前市场上 Agent 相关的岗位类型：

┌──────────────────┬─────────────────────────────────────────┐
│      岗位         │             工作内容与要求                │
├──────────────────┼─────────────────────────────────────────┤
│ Agent 应用工程师  │ 用 LangChain/LangGraph 构建 Agent 应用    │
│                  │ 要求：Python + 框架熟练 + 业务理解力       │
├──────────────────┼─────────────────────────────────────────┤
│ Agent 平台工程师  │ 开发 Agent 基础设施（编排/评测/监控）      │
│                  │ 要求：Python/Go + 分布式系统 + 工程能力    │
├──────────────────┼─────────────────────────────────────────┤
│ Agent 产品经理    │ 定义 Agent 产品形态和用户体验             │
│                  │ 要求：产品感 + Agent 技术理解 + 数据分析   │
├──────────────────┼─────────────────────────────────────────┤
│ Prompt 工程师     │ 专门设计和优化 Agent 的提示词             │
│                  │ 要求：语言学功底 + 系统思维 + 评测经验     │
├──────────────────┼─────────────────────────────────────────┤
│ Agent 研究员      │ Agent 学术研究（新架构/新范式）           │
│                  │ 要求：顶会论文 + 理论基础 + 工程实现能力   │
└──────────────────┴─────────────────────────────────────────┘

招聘趋势（2024-2025）：
  - Agent 应用工程师需求量最大（占 60%+）
  - 岗位从大厂向中厂/创业公司扩散
  - 薪资范围：一线城市 25K-60K（视经验和公司而定）
  - 关键技能排序：LangChain/LangGraph > Python > LLM 原理 > 系统设计

主要雇主类型：
  - 大厂：字节跳动(Coze)、阿里(Tongyi Agent)、腾讯
  - Agent 创业公司：Dify、扣子、澜码科技
  - AI 平台公司：智谱AI、百川智能、月之暗面
  - 传统企业 AI 部门：金融、电商、教育等行业
"""


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7.2 面试高频问题 TOP 20
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

每道题都附「回答框架」和「得分点」。
"""

AGENT_INTERVIEW_QUESTIONS = [
    # ===== 基础概念篇（必考 5 题）=====
    {
        "id": 1,
        "question": "什么是 AI Agent？它和普通 LLM 调用有什么区别？",
        "category": "基础概念",
        "framework": (
            "1. 定义：Agent = LLM + 规划 + 记忆 + 工具\n"
            "2. 区别：LLM 只能生成文本，Agent 能自主决策、调用工具、执行行动\n"
            "3. 举例：ChatGPT 只能回答问题，AutoGPT 能自己搜索+计算+总结\n"
            "4. 核心循环：Perceive → Think → Act → Observe"
        ),
        "scoring_points": [
            "提到「Agent = LLM + 规划 + 记忆 + 工具」这个公式",
            "能描述 Perceive-Think-Act-Observe 循环",
            "给出具体的对比示例",
        ],
    },
    {
        "id": 2,
        "question": "解释一下 ReAct 模式，它在 Agent 中如何工作？",
        "category": "基础概念",
        "framework": (
            "1. ReAct = Reasoning + Acting\n"
            "2. 流程：Thought → Action → Observation → Thought → ...\n"
            "3. 每次循环：LLM 先推理(Thought)，然后决定行动(Action)，"
            "根据结果(Observation)继续推理\n"
            "4. 优点：灵活、通用；缺点：每步都调 LLM，成本高"
        ),
        "scoring_points": [
            "能画出 ReAct 循环的流程图",
            "提到 ReAct 论文(Yao et al., ICLR 2023)",
            "能对比 ReAct 和其他模式的优劣",
        ],
    },
    {
        "id": 3,
        "question": "Agent 的记忆系统有哪些类型？各自怎么实现？",
        "category": "基础概念",
        "framework": (
            "1. 短期记忆：对话上下文 → messages 列表 → 受上下文窗口限制\n"
            "2. 长期记忆：跨会话持久化 → 向量数据库(ChromaDB/Pinecone) → RAG 检索\n"
            "3. 工作记忆：当前任务中间状态 → Python 变量/缓存 → 任务完成后释放\n"
            "4. 面试加分：提及滑动窗口、摘要压缩、混合检索等上下文管理策略"
        ),
        "scoring_points": [
            "三种记忆类型都说全",
            "给出具体的实现方案（不是抽象的）",
            "提到上下文窗口限制的解决方案",
        ],
    },
    {
        "id": 4,
        "question": "Function Calling 的原理是什么？LLM 如何知道该调用哪个函数？",
        "category": "基础概念",
        "framework": (
            "1. Function Calling 是 LLM 的结构化输出能力\n"
            "2. 我们预定义工具的 JSON Schema（名称、描述、参数）\n"
            "3. LLM 推理后输出一个 JSON 对象 {\"name\": \"...\", \"arguments\": {...}}\n"
            "4. 我们的代码解析这个 JSON 并执行对应的函数\n"
            "5. 将执行结果返回给 LLM，让它继续推理"
        ),
        "scoring_points": [
            "理解工具描述(description)是 LLM 选工具的唯一依据",
            "说明 Function Calling 是结构化输出，不是 LLM 真的「调用」函数",
            "提到完整的调用-执行-返回循环",
        ],
    },
    {
        "id": 5,
        "question": "Agent 和 RAG 是什么关系？什么场景下分别使用？",
        "category": "基础概念",
        "framework": (
            "1. RAG = 检索增强生成：从知识库检索相关信息 + LLM 生成回答\n"
            "2. Agent 包含 RAG：RAG 可以作为 Agent 的一个工具\n"
            "3. 简单问答用 RAG，需要多步骤推理/工具调用用 Agent\n"
            "4. RAG 是 Agent 记忆系统（长期记忆）的常用实现方式"
        ),
        "scoring_points": [
            "明确 RAG 和 Agent 不是互斥的，是包含关系",
            "给出选择 RAG vs Agent 的判断标准",
            "能举例说明融合使用的场景",
        ],
    },

    # ===== 框架与工程篇（必考 5 题）=====
    {
        "id": 6,
        "question": "LangChain Agent 的核心架构是什么？和裸写有什么区别？",
        "category": "框架工程",
        "framework": (
            "1. 核心：create_react_agent(LLM, tools, checkpointer)\n"
            "2. 对比裸写：自动管理 messages、自动处理 tool_calls 循环、"
            "内置记忆管理\n"
            "3. LangChain 提供了 Tool/Toolkit/AgentExecutor 等抽象层\n"
            "4. 优势是开发效率，劣势是多了一层抽象（调试困难）"
        ),
        "scoring_points": [
            "能说出 LangChain 帮我们做的 3 件事",
            "理解抽象层的利弊",
            "提到自己手写过 Agent（说明懂底层原理）",
        ],
    },
    {
        "id": 7,
        "question": "LangGraph 是什么？它和 LangChain Agent 有什么区别？",
        "category": "框架工程",
        "framework": (
            "1. LangGraph = 基于状态机的 Agent 编排框架\n"
            "2. 核心概念：State(共享状态) + Node(执行节点) + Edge(流转控制)\n"
            "3. LangChain Agent 是固定的 ReAct 循环\n"
            "4. LangGraph 可以自定义任意复杂的执行图（分支/循环/并行）\n"
            "5. 生产环境建议用 LangGraph（更可控）"
        ),
        "scoring_points": [
            "解释 State-Node-Edge 三要素",
            "举例说明什么场景需要自定义执行图",
            "提到 LangGraph 在生产环境的优势",
        ],
    },
    {
        "id": 8,
        "question": "你在实际项目中遇到过什么 Agent 的困难？怎么解决的？",
        "category": "框架工程",
        "framework": (
            "准备 2-3 个真实的踩坑经历：\n"
            "1. LLM 幻觉导致工具参数错误 → 增加参数校验 + enum 限制\n"
            "2. 上下文过长超出限制 → 滑动窗口 + 摘要压缩\n"
            "3. Agent 陷入死循环 → 设置 max_iterations + 检测重复调用\n"
            "4. 用具体数字说明效果（错误率从 X% 降到 Y%）"
        ),
        "scoring_points": [
            "问题描述具体（不是泛泛而谈）",
            "有数据对比（修复前/后）",
            "展现问题分析和解决能力",
        ],
    },
    {
        "id": 9,
        "question": "多 Agent 系统怎么设计？Agent 之间怎么通信？",
        "category": "框架工程",
        "framework": (
            "1. 三种架构：协作式、分层式、竞争式\n"
            "2. 通信方式：\n"
            "   - 共享状态（LangGraph）：通过 TypedDict 共享数据\n"
            "   - 消息传递（AutoGen）：Agent 之间发消息对话\n"
            "   - 任务委派（crewAI）：上级分配任务给下级\n"
            "3. 关键设计：职责单一、接口标准化、错误隔离"
        ),
        "scoring_points": [
            "三种架构都说全",
            "能对比三种通信方式的优劣",
            "提到错误隔离和成本控制",
        ],
    },
    {
        "id": 10,
        "question": "生产环境的 Agent 需要注意什么？",
        "category": "框架工程",
        "framework": (
            "1. 可观测性：Tracing(LangSmith) + Metrics + Alerting\n"
            "2. 成本控制：模型分层、缓存、流式输出\n"
            "3. 安全：Prompt Injection 防护、权限最小化、人在回路\n"
            "4. 可靠性：重试机制、超时控制、降级策略\n"
            "5. 性能：工具并行调用、长连接复用"
        ),
        "scoring_points": [
            "不只说功能，说运维和生产质量",
            "提到具体的工具和策略",
            "展现了「工程化思维」而非「Demo 思维」",
        ],
    },

    # ===== 系统设计篇（必考 5 题）=====
    {
        "id": 11,
        "question": "设计一个智能客服 Agent，你会怎么做？",
        "category": "系统设计",
        "framework": (
            "1. 需求分析：用户意图识别、FAQ 匹配、工单创建、人工转接\n"
            "2. 工具设计：知识库检索(RAG)、订单查询API、工单系统API\n"
            "3. Agent 架构：\n"
            "   - 路由层：意图识别 Agent 分发到不同子 Agent\n"
            "   - 执行层：FAQ Agent / 订单 Agent / 投诉 Agent\n"
            "   - 记忆：对话历史 + 用户画像 + 知识库\n"
            "4. 评估：解决率、响应时间、用户满意度"
        ),
        "scoring_points": [
            "体现了分层架构设计",
            "考虑了人机协作（转人工）",
            "定义了明确的评估指标",
        ],
    },
    {
        "id": 12,
        "question": "Agent 的上下文窗口只有 128K，如何处理超长对话？",
        "category": "系统设计",
        "framework": (
            "1. 滑动窗口：只保留最近 N 条消息\n"
            "2. 摘要压缩：将早期对话用 LLM 总结为摘要\n"
            "3. 向量检索：将历史存入向量库，按相关性检索\n"
            "4. 分层记忆：近期用全量，中期用摘要，远期用检索\n"
            "5. 混合策略：滑动窗口(最新) + 摘要(中期) + 向量检索(远期)"
        ),
        "scoring_points": [
            "不只说摘要压缩，提到多种策略",
            "能解释混合策略的优势",
            "提到向量数据库的具体实现",
        ],
    },
    {
        "id": 13,
        "question": "如何保证 Agent 的工具调用安全性？",
        "category": "系统设计",
        "framework": (
            "1. 权限最小化：每个 Agent 只能访问必要的工具\n"
            "2. 输入校验：参数类型检查 + 值域限制(enum) + 正则过滤\n"
            "3. 操作分级：读操作自动执行，写操作需人工确认\n"
            "4. Prompt Injection 防护：\n"
            "   - 用户输入和系统指令用分隔符隔离\n"
            "   - 对用户输入做消毒处理\n"
            "5. 审计日志：记录所有工具调用"
        ),
        "scoring_points": [
            "提到 Prompt Injection 及其防护",
            "理解「读自动、写入确认」的分级策略",
            "提到审计和可追溯性",
        ],
    },
    {
        "id": 14,
        "question": "怎么降低 Agent 的 API 调用成本？",
        "category": "系统设计",
        "framework": (
            "1. 模型分层：简单任务用小模型(gpt-4o-mini)，复杂任务用大模型\n"
            "2. 语义缓存：相同问题缓存结果，避免重复调用\n"
            "3. 减少 round-trip：一次 prompt 解决多个问题\n"
            "4. Batching：将多个独立工具调用合并\n"
            "5. 响应长度控制：合理设置 max_tokens"
        ),
        "scoring_points": [
            "不只说「换个便宜模型」",
            "提到语义缓存的实现思路",
            "给出量化的降本预测(如从 $X 降到 $Y)",
        ],
    },
    {
        "id": 15,
        "question": "如何评估 Agent 的表现？怎么建立评测体系？",
        "category": "系统设计",
        "framework": (
            "1. 多维度：任务完成率、工具准确率、执行效率、用户满意度\n"
            "2. 评测方法：LLM-as-Judge + 人工标注 + A/B 测试\n"
            "3. 评测集建设：覆盖典型场景 + 边界情况 + Bad Case 回归\n"
            "4. 持续监控：线上指标看板 + 告警阈值 + 定期巡检"
        ),
        "scoring_points": [
            "提到 LLM-as-Judge 方法",
            "懂得评测不是一次性的而是持续的",
            "提到 Bad Case 回归机制",
        ],
    },

    # ===== 开放讨论篇（必考 5 题）=====
    {
        "id": 16,
        "question": "Agent 当前最大的技术挑战是什么？",
        "category": "开放讨论",
        "framework": (
            "1. 可靠性：LLM 的不确定性导致 Agent 行为不可预测\n"
            "2. 规划能力：复杂任务的分解和编排还不成熟\n"
            "3. 执行效率：多步推理的延迟和成本\n"
            "4. 安全性：Prompt Injection 和权限控制\n"
            "5. 评估困难：缺乏标准化的 Agent 评测体系\n"
            "任选 2-3 个深入展开，要有自己的见解"
        ),
        "scoring_points": [
            "展现对行业痛点的理解深度",
            "有自己的独立见解而非人云亦云",
            "能联系自己的实践经验",
        ],
    },
    {
        "id": 17,
        "question": "Agent 的未来发展趋势是什么？",
        "category": "开放讨论",
        "framework": (
            "1. 从单 Agent 到 Multi-Agent 协作\n"
            "2. 从文本到多模态（视觉Agent、操作Agent）\n"
            "3. 从辅助到自主（Computer Use、代码自主修复）\n"
            "4. 从通用到垂直（金融Agent、医疗Agent、法律Agent）\n"
            "5. Agent 基础设施成熟（评测、监控、安全）"
        ),
        "scoring_points": [
            "观点有逻辑支撑",
            "关注行业前沿（提到 Claude Computer Use 等）",
            "能和自己的职业规划联系起来",
        ],
    },
    {
        "id": 18,
        "question": "你最近关注了哪些 Agent 相关的论文/开源项目？",
        "category": "开放讨论",
        "framework": (
            "必读论文(面试前至少读过3篇):\n"
            "- ReAct (Yao et al., 2023)\n"
            "- Reflexion (Shinn et al., 2023)\n"
            "- AutoGPT / SWE-Agent / Devin\n"
            "\n"
            "必知项目:\n"
            "- LangChain / LangGraph\n"
            "- crewAI / AutoGen\n"
            "- Coze / Dify\n"
            "- Anthropic Computer Use\n"
            "\n"
            "回答时要能说出「这个工作的核心贡献是什么」"
        ),
        "scoring_points": [
            "能说出 2-3 篇论文的核心贡献",
            "不只是「知道」而是「读过并理解」",
            "能评论论文的优缺点",
        ],
    },
    {
        "id": 19,
        "question": "给你一个想法，你如何从零构建一个 Agent 产品？",
        "category": "开放讨论",
        "framework": (
            "1. 需求定义：用户是谁？解决什么问题？核心场景是什么？\n"
            "2. 技术选型：裸写 vs LangChain vs LangGraph？用什么模型？\n"
            "3. MVP 开发：最小可验证产品，1-2 周出 demo\n"
            "4. 评测迭代：收集反馈 → 改进 prompt → 调整工具 → 迭代\n"
            "5. 生产化：监控、安全、性能优化、成本控制"
        ),
        "scoring_points": [
            "展现产品思维（不是纯技术视角）",
            "有清晰的阶段划分和时间预期",
            "提到迭代和反馈循环",
        ],
    },
    {
        "id": 20,
        "question": "你有什么想问我们的？",
        "category": "开放讨论",
        "framework": (
            "推荐问的问题（展现你的深度）:\n"
            "- 团队目前用的是什么 Agent 架构？为什么选它？\n"
            "- Agent 在落地过程中最大的挑战是什么？\n"
            "- 团队对 Agent 评测是怎么做的？\n"
            "- Agent 产品的下一步规划是什么？\n"
            "\n"
            "避免问:\n"
            "- 薪资福利（留给 HR 面）\n"
            "- 公司是做什么的（说明你没做功课）"
        ),
        "scoring_points": [
            "问题展现你对 Agent 技术的深入理解",
            "体现你对这份工作的认真态度",
        ],
    },
]


def print_interview_questions():
    """打印所有高频面试题的分类摘要。"""
    categories = {}
    for q in AGENT_INTERVIEW_QUESTIONS:
        cat = q["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(q)

    for cat, questions in categories.items():
        print(f"\n{'='*60}")
        print(f"  📋 {cat} ({len(questions)} 题)")
        print(f"{'='*60}")
        for q in questions:
            print(f"\n  Q{q['id']}: {q['question']}")
            print(f"  回答框架: {q['framework'][:120]}...")
            print(f"  得分点: {', '.join(q['scoring_points'][:2])}...")


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7.3 项目准备指南
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

面试中最有说服力的永远是你的项目经验。
以下是推荐的 Agent 项目组合：

必做项目（基础)：
  1. 智能助手 Agent
     - 功能：搜索 + 计算 + 天气 + 新闻
     - 技术：LangChain + ReAct Agent
     - 亮点：展示对 Agent 基本架构的理解

  2. 文档分析 Agent
     - 功能：上传文档 → 提取关键信息 → 生成摘要 → 回答提问
     - 技术：RAG + Agent + 向量数据库
     - 亮点：展示记忆系统和工具调用的整合

进阶项目（加分项）:
  3. 代码审查 Agent
     - 功能：Read PR → Static Analysis → Code Review → 生成 Review 报告
     - 技术：LangGraph + Git API + 代码分析工具
     - 亮点：展示复杂工作流编排能力

  4. 数据分析 Agent
     - 功能：自然语言提问 → 生成 SQL → 执行查询 → 可视化
     - 技术：Text-to-SQL + Pandas + Plotly
     - 亮点：展示多工具协作 + 结构化输出

  5. Multi-Agent 内容工厂
     - 功能：策划 → 调研 → 写作 → 编辑 → 配图 → 发布
     - 技术：crewAI / LangGraph Multi-Agent
     - 亮点：展示多 Agent 架构设计能力

项目展示技巧：
  1. 准备一个 3 分钟的「Elevator Pitch」
  2. 提前画好架构图（面试时分享屏幕/画白板）
  3. 准备 1-2 个「技术难点 + 解决方案」的故事
  4. 有性能/成本的量化数据最好（延迟降到 Xms、成本降低 Y%）
"""


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7.4 面试流程与时间线
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

典型 Agent 工程师面试流程（大厂）:

  第 1 轮：技术电话面试 (45 分钟)
    - 基础概念考核（本章 7.2 的 Q1-Q5）
    - 一道简单的手写代码（如工具定义、API 调用）

  第 2 轮：在线编码 (60 分钟)
    - 实现一个简易 Agent 循环
    - 设计工具和评估方案

  第 3 轮：系统设计 (60 分钟)
    - 设计一个特定场景的 Agent 系统
    - 考量架构合理性、扩展性、安全性

  第 4 轮：项目深挖 (45 分钟)
    - 详细介绍一个你做过的 Agent 项目
    - 追问细节：为什么这么做？遇到什么问题？

  第 5 轮：行为面试 (45 分钟)
    - 团队协作、学习能力、职业规划
    - STAR 法则回答（Situation-Task-Action-Result）

准备时间线：
  - 面试前 2 周：回顾第 1-3 章（基础概念）
  - 面试前 1 周：复习第 4-5 章（框架实战）+ 准备项目故事
  - 面试前 3 天：过一遍 7.2 的所有问题
  - 面试前 1 天：Mock Interview + 休息
"""


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7.5 课程总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

回顾整个学习路线：

  Ch0 概览        → 理解 Agent 是什么，搭建环境
  Ch1 基础        → 手写第一个 Agent，理解 ReAct 循环
  Ch2 组件        → 深入规划器、记忆系统、工具设计
  Ch3 类型        → ReAct / Plan-Execute / Reflexion 对比
  Ch4 框架        → LangChain Agent + LangGraph 实战
  Ch5 多智能体    → Multi-Agent 架构与协作模式
  Ch6 评估        → 评测体系、测试策略、生产实践
  Ch7 求职        → 面试问题、项目准备、行业趋势 ← 你在这里

下一步行动建议：
  1. 动手做一个完整的 Agent 项目（从本章 7.3 选题）
  2. 把项目部署上线（Streamlit / Gradio / FastAPI）
  3. 写一篇技术博客分享你的项目（展示技术影响力）
  4. 参加 Agent 相关的 Hackathon
  5. 关注 arXiv 上最新的 Agent 论文
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║    第7章：Agent 求职面试准备与行业实践                 ║")
    print("║    高频面试题 · 项目指南 · 面试流程                   ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n▶ 7.1 Agent 岗位全景图")
    print("-" * 50)
    roles = [
        "Agent 应用工程师 → 需求量最大(60%+) → 25K-60K",
        "Agent 平台工程师 → 要求分布式系统能力",
        "Agent 产品经理   → 产品感 + 技术理解",
        "Prompt 工程师    → 系统化提示词设计",
        "Agent 研究员     → 顶会论文 + 新范式探索",
    ]
    for r in roles:
        print(f"  • {r}")

    print("\n▶ 7.2 高频面试题 TOP 20（摘要）")
    print_interview_questions()

    print("\n\n▶ 7.3 项目准备推荐")
    print("-" * 50)
    projects = [
        "必做: 智能助手 Agent (LangChain + ReAct)",
        "必做: 文档分析 Agent (RAG + Agent)",
        "进阶: 代码审查 Agent (LangGraph + Git API)",
        "进阶: 数据分析 Agent (Text-to-SQL + 可视化)",
        "进阶: Multi-Agent 内容工厂 (crewAI/LangGraph)",
    ]
    for p in projects:
        print(f"  • {p}")

    print("\n▶ 7.4 面试流程")
    print("-" * 50)
    rounds = [
        "第1轮: 技术电话面(45min) - 基础概念 + 手写代码",
        "第2轮: 在线编码(60min)   - 实现 Agent 循环",
        "第3轮: 系统设计(60min)   - Agent 系统架构",
        "第4轮: 项目深挖(45min)   - 你的 Agent 项目",
        "第5轮: 行为面试(45min)   - STAR 法则",
    ]
    for r in rounds:
        print(f"  • {r}")

    print("\n\n" + "=" * 60)
    print("  🎓 恭喜完成全部课程！")
    print("  ")
    print("  从 Agent 基础概念到面试求职，你现在已经具备了")
    print("  成为一名 AI Agent 工程师所需的核心知识体系。")
    print("  ")
    print("  下一步：动手做一个项目，把它展示给面试官。")
    print("=" * 60)
