"""
第8章：Claude Code 架构深度剖析 —— 工业级 Agent 的标杆实现
===========================================================

📌 本章目标：
  1. 深入理解 Claude Code 的完整系统架构（4层结构）
  2. 掌握主 Agent 循环 (nO) 的设计哲学 —— 「用简单对抗复杂」
  3. 理解 h2A 实时 Steering 机制的工作原理
  4. 掌握分层 Multi-Agent 架构：Task 工具 → SubAgent 创建 → 并发调度
  5. 理解 Context Compaction（上下文压缩）的触发与执行机制
  6. 了解 Claude Agent SDK（原 Claude Code SDK）的设计理念

📌 面试高频点：
  - Claude Code 的 Agent 循环和 LangChain 的 ReAct 循环有何不同？
  - h2A 异步双缓冲队列是如何实现「中途介入」的？
  - 为什么 Claude Code 选择单线程主循环而非并行多 Agent？
  - Context Compaction 在什么时机触发？具体做了什么？
  - SubAgent 和主 Agent 之间的隔离是如何实现的？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本章基于 Claude Code v1.0.33 的社区逆向工程分析结果
（来源：shareAI-lab/analysis_claude_code + 官方 SDK 文档）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


8.1 Claude Code 的整体架构 —— 4 层分层设计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Claude Code 的架构可以抽象为 4 个层次：

┌─────────────────────────────────────────────────────┐
│  Layer 1: 用户交互层 (User Interaction Layer)         │
│  ├─ CLI 界面 (REPL)                                  │
│  ├─ VS Code 插件                                      │
│  ├─ Web UI                                           │
│  └─ Claude Agent SDK (供开发者嵌入)                   │
├─────────────────────────────────────────────────────┤
│  Layer 2: Agent 核心调度层 (Core Scheduling Layer)    │
│  ├─ nO: 主 Agent 循环引擎（单线程）                    │
│  ├─ h2A: 异步双缓冲消息队列（实时 Steering）          │
│  ├─ StreamGen: 流式输出管理                           │
│  ├─ ToolEngine: 工具调用编排器                        │
│  └─ Compressor (wU2): 上下文压缩器                    │
├─────────────────────────────────────────────────────┤
│  Layer 3: 工具执行与管理层 (Tool Execution Layer)     │
│  ├─ Bash / Grep / View / Edit / Write 等基础工具       │
│  ├─ Task 工具 → SubAgent 创建与管理                   │
│  ├─ MCP 工具集成（连接外部 MCP Server）               │
│  └─ 权限与安全控制 (Permission System)                │
├─────────────────────────────────────────────────────┤
│  Layer 4: 存储与持久化层 (Storage & Persistence)      │
│  ├─ TODO 列表（短期任务管理）                          │
│  ├─ Markdown 文件（长期项目记忆）                      │
│  ├─ Git 集成（diff-based 工作流）                     │
│  └─ 会话状态持久化                                    │
└─────────────────────────────────────────────────────┘

核心理念：「做简单的事」—— Markdown 代替数据库、正则代替向量搜索、
        单线程代替并行、平面消息代替复杂线程。


8.2 主 Agent 循环 (nO) —— 用简单对抗复杂
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

nO 是 Claude Code 的心脏。它实现了一个极其简洁的 while 循环：

   while (response has tool_calls):
       execute tools in parallel (when safe)
       feed results back to model
   return final text response

这和我们第1章手写的 Agent 循环几乎一模一样！

但 Claude Code 的设计哲学值得深思：

1. 「单线程主循环」—— 而不是多 Agent 同时跑
   - 原因：可调试性 > 并发性能
   - 一个线程、一个消息列表、一条执行路径
   - 出问题时，你可以从头到尾跟踪每一步

2. 「平面消息列表」—— 而不是复杂的线程/层级结构
   - 原因：LLM 对平面结构的理解最稳定
   - 不构建复杂的树形消息结构

3. 「做最简单的事」—— 始终选择实现成本最低的方案
   - 搜索代码用 Grep（正则）而不是 Embedding（向量）
   - 存记忆用 Markdown 文件而不是数据库
   - 这些选择使系统更易理解、更易调试

核心代码概念（伪代码还原）：

    # nO 主循环的简化实现
    async def agent_loop_nO(user_input: str, context: dict):
        messages = [build_system_prompt(), {"role": "user", "content": user_input}]

        while True:
            # 检查是否需要上下文压缩
            if estimate_tokens(messages) > THRESHOLD_PCT * MAX_TOKENS:
                messages = await compressor_wU2(messages)

            # 调用 LLM
            response = await llm.beta.messages.create(
                model="claude-sonnet-4-20250514",
                messages=messages,
                tools=TOOLS,
                max_tokens=8192,
            )

            # 如果 LLM 直接返回文本（无工具调用），退出循环
            if not response.tool_calls:
                return response.content

            # 执行工具调用（可并行）
            tool_results = await execute_tools(response.tool_calls)

            # 将 assistant 消息 + tool 结果加入消息列表
            messages.append({"role": "assistant", "content": response})
            for result in tool_results:
                messages.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": result.id,
                    "content": result.output,
                }]})


8.3 h2A 实时 Steering 机制 —— 让人可以「中途介入」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是 Claude Code 最被低估的创新。

大多数 Agent 的工作模式是「提交任务 → 等待完成」。
如果你在 Agent 执行到一半时想修正方向，唯一的选择是中断并重新开始。

h2A 改变了这一切。

技术原理：
  1. h2A 是一个「异步双缓冲消息队列」
  2. 主循环 nO 在等待 LLM 响应时，h2A 可以接收用户的新消息
  3. 当 nO 完成当前步骤时，检查 h2A 中是否有新的用户指令
  4. 如果有，将新指令注入对话，Agent 无缝调整方向

类比：
  你是一个正在解题的学生，老师可以在不打断你的情况下
  递给你一张纸条：「注意第3题的公式用错了」。
  你看到纸条后，立即调整思路，继续往下做。

伪代码实现：

    # h2A 双缓冲队列的简化实现
    class h2AQueue:
        def __init__(self):
            self._active_buffer = []
            self._pending_buffer = []
            self._lock = asyncio.Lock()

        async def push_user_message(self, msg):
            # 用户中途发送消息（非阻塞）
            async with self._lock:
                self._pending_buffer.append(msg)

        async def drain_pending(self):
            # nO 在每步结束后消费待处理消息
            async with self._lock:
                # 原子性地交换两个缓冲区
                self._active_buffer, self._pending_buffer = self._pending_buffer, []
            msgs = self._active_buffer
            return msgs

    # nO 循环中使用 h2A
    async def agent_loop_with_steering(user_input, h2a_queue):
        messages = [...]
        while True:
            # 第一步：检查是否有用户中途介入
            steering_msgs = await h2a_queue.drain_pending()
            if steering_msgs:
                messages.append({"role": "user", "content": steering_msgs[0]})
                # 不需要重新开始，只需在现有对话中追加即可

            response = await llm.create(messages=messages, tools=TOOLS)
            if not response.tool_calls:
                return response.content
            # 执行工具...
            await execute_tools(response.tool_calls)

关键数据（来自逆向工程分析）：
  - 吞吐量 > 10,000 消息/秒
  - 零延迟消息传递
  - Promise-based 异步迭代器 + 智能背压控制
"""

import asyncio
from typing import Optional


class SimulatedSteeringQueue:
    """模拟 Claude Code 的 h2A 异步双缓冲队列。

    演示如何在不中断 Agent 的情况下接受用户中途介入。
    """

    def __init__(self):
        self._pending = []
        self._lock = asyncio.Lock()

    async def inject(self, message: str):
        """用户中途注入新指令（非阻塞）。"""
        async with self._lock:
            self._pending.append(message)

    async def drain(self) -> list[str]:
        """Agent 消费所有待处理的注入消息。"""
        async with self._lock:
            msgs = self._pending.copy()
            self._pending.clear()
        return msgs


"""
8.4 上下文压缩 (Context Compaction) —— 解决长对话难题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是 Claude Code 能够在数小时的长任务中保持稳定的关键技术。

核心机制：
  1. Compressor (代号 wU2) 监控消息列表的 token 使用量
  2. 当使用量达到上下文窗口的 ~92% 时自动触发
  3. Compressor 调用一次 LLM，生成「对话摘要」
  4. 将早期消息（已总结的）移除，替换为摘要
  5. 将重要信息移动到 Markdown 文件（长期记忆）

为什么是 92%？
  - 太早压缩 → 浪费计算资源
  - 太晚压缩 → 没有空间放 tool 返回结果
  - 92% 是经过工程实践的平衡点

为什么用 Markdown 而不是专用数据库？
  - Markdown 对 LLM 最友好（训练数据中大量存在）
  - 人类可读，便于调试
  - 零依赖，无需额外基础设施

压缩过程示意：
  压缩前：[System][User][Asst][Tool][Asst][Tool][User][Asst]...
                              ↑ 92% 阈值
  压缩后：[System][Summary][最近3轮对话，后续消息]

  被压缩的内容：
    → 生成简短摘要（放入 Summary）
    → 重要信息写入 Markdown 长期记忆文件
    → 不重要的细节直接丢弃
"""


def simulate_context_compaction():
    """模拟上下文压缩的过程。"""
    print("=" * 60)
    print("  Context Compaction 模拟")
    print("=" * 60)

    MAX_TOKENS = 200000  # Claude 的上下文窗口
    THRESHOLD = 0.92     # 触发压缩的阈值

    # 模拟一个不断增长的对话历史
    conversation = [
        {"role": "system", "content": "你是一个代码助手..."},
        {"role": "user", "content": "帮我重构整个项目..."},
        {"role": "assistant", "content": "好的，让我先分析项目结构..."},
    ]

    # 模拟大量工具调用和结果
    for i in range(20):
        conversation.append({
            "role": "assistant",
            "content": f"调用工具以处理文件 {i}...",
        })
        conversation.append({
            "role": "tool",
            "content": f"文件 {i} 的处理结果：成功修改 50 行",
        })

    current_tokens = len(str(conversation)) // 4  # 粗略估算
    usage_pct = current_tokens / MAX_TOKENS

    print(f"  当前 token 估算: {current_tokens}")
    print(f"  使用率: {usage_pct:.1%}")
    print(f"  阈值: {THRESHOLD:.0%}")

    if usage_pct > THRESHOLD:
        print(f"\n  ⚡ 触发压缩！")
        # 压缩策略：保留前 2 条（System + 第一条 user），
        #          中间生成摘要，保留最后 6 条（最近3轮）
        system = conversation[0:2]
        recent = conversation[-6:]
        middle = conversation[2:-6]

        summary = f"[压缩摘要] 此前共 {len(middle)} 条消息，" \
                  f"主要内容：分析了项目结构，处理了多个文件..."

        compacted = system + [
            {"role": "system", "content": summary}
        ] + recent

        new_tokens = len(str(compacted)) // 4
        new_usage = new_tokens / MAX_TOKENS
        print(f"  压缩后 token 估算: {new_tokens}")
        print(f"  压缩后使用率: {new_usage:.1%}")
        print(f"  节省: {(current_tokens - new_tokens) / current_tokens:.0%}")


"""
8.5 分层 Multi-Agent 架构 —— Task 工具与 SubAgent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Claude Code 的多 Agent 是通过「Task 工具」实现的：
  - 主 Agent 调用 Task 工具 → 创建 SubAgent
  - SubAgent 在独立上下文中运行
  - SubAgent 完成后返回结果
  - 主 Agent 合成结果

关键设计（Claude Code 1.0.60+ 版本）：

1. 上下文过滤（Context Filtering）
   - v1.0.59: SubAgent 继承全部对话历史（浪费大量 token）
   - v1.0.60+: SubAgent 只获得与任务相关的精简上下文
   - 效果：Token 消耗降低约 70%

2. 工具权限控制
   - 研究型 SubAgent：只读权限（Grep, View）
   - 实施型 SubAgent：完整权限（Edit, Write, Bash）
   - 权限通过 SubAgent 创建时动态配置

3. 并发限制
   - 最多同时运行一个 SubAgent（v1.0.33 时期）
   - 这不是技术限制，而是设计选择
   - 原因：避免 Agent 失控，保持可预测性

伪代码还原：

    # Task 工具背后的 SubAgent 创建与执行
    def create_subagent(task_description: str, task_prompt: str,
                        available_tools: list, permissions: dict):
        # 1. 构建 SubAgent 专属系统提示词
        system_prompt = (
            "You are a specialized sub-agent.\n"
            f"Task: {task_description}\n"
            f"You have access to tools: "
            f"{[t.name for t in available_tools]}\n"
            f"Your permissions: {permissions}\n\n"
            "IMPORTANT:\n"
            "- You are stateless. Complete the task in one response.\n"
            "- Return your findings concisely.\n"
            "- Do not ask follow-up questions."
        )

        # 2. 执行 SubAgent（独立对话）
        sub_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        # 3. SubAgent 执行自己的循环
        result = run_agent_loop(sub_messages, tools=available_tools)

        # 4. 将结果返回主 Agent
        return {"sub_agent_result": result}

Comparision with crewAI:
  ┌──────────────────┬───────────────────┬──────────────────────┐
  │      特性         │    Claude Code     │       crewAI          │
  ├──────────────────┼───────────────────┼──────────────────────┤
  │ SubAgent 创建     │ Task 工具动态创建  │ 预定义 Agent 角色      │
  │ 上下文隔离        │ 完全独立           │ 共享部分上下文         │
  │ 状态管理          │ 无状态（单次执行）  │ 有状态（多次交互）     │
  │ 并发              │ 受控（有限并发）    │ 自由并发              │
  │ 适用场景          │ 编程、搜索、分析    │ 内容创作、多角色协作   │
  └──────────────────┴───────────────────┴──────────────────────┘


8.6 Claude Agent SDK —— Agent 开发的新范式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2025年9月，Anthropic 将 Claude Code SDK 重命名为 Claude Agent SDK。

核心理念：「给 Claude 一台电脑」

  Claude Code 之所以强大，不是因为特殊的 Agent 架构，
  而是因为它给了 Claude 程序员每天使用的工具：
  - 文件系统（读写文件）
  - 终端（执行命令）
  - 代码搜索（Grep）
  - 代码编辑（Edit）

  这些都是通用能力，不限于编程。
  Claude 可以用来读 CSV、搜索网页、生成图表、管理日历...

SDK 的 Agent 循环设计：
  Gather Context → Take Action → Verify Work → Repeat

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ 收集上下文    │────→│  执行行动     │────→│  验证结果     │
  │ - 搜索文件    │     │ - 编辑/写入   │     │ - 运行测试    │
  │ - 读取代码    │     │ - 运行命令    │     │ - 检查 lint   │
  │ - 查询历史    │     │ - 调用 API    │     │ - 审查 diff   │
  └──────────────┘     └──────────────┘     └──────────────┘
         ↑                                        │
         └────────────────────────────────────────┘
                        循环直到完成


8.7 Claude Code 的设计原则总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

从 Claude Code 可以学到的 Agent 工程哲学：

1. Simple is Better
   - 单线程主循环 > 复杂的并行调度
   - 平面消息列表 > 树形消息结构
   - Markdown 文件 > 专用数据库

2. Tool-First Thinking
   - Agent 的能力边界 = 工具的能力边界
   - 好工具胜过好 prompt

3. Controllable Autonomy（可控自主）
   - 不追求完全自主，追求可控的自主
   - 人在回路 (h2A Steering)
   - 权限分级的工具访问

4. Context Engineering（上下文工程）
   - 上下文压缩是 Agent 稳定性的关键
   - 92% 阈值是工程实践的产物
   - Markdown 长期记忆是最简单的持久化方案

面试速记：
  "Claude Code 的架构有什么特别之处？"
  → 简洁的主循环 + 实时 Steering(h2A) + 上下文压缩 + 分层 SubAgent
  → 设计哲学：做简单的事，用简单的工具
  → 不给 LLM 过多选择，控制在可预测的路径上


8.8 本章总结
━━━━━━━━━━━━━━

核心要点回顾：

1. 4层架构：用户交互 → Agent 核心调度 → 工具执行 → 存储持久化

2. nO 主循环：简洁的 while(tool_calls) 循环，单线程，平面消息

3. h2A Steering：异步双缓冲队列，允许用户中途介入不中断任务

4. Context Compaction：92% 阈值触发，压缩早期对话为摘要

5. SubAgent：通过 Task 工具创建，独立上下文，1.0.60 引入上下文过滤

6. Claude Agent SDK：给 Claude 一台电脑，构建通用 Agent

面试常问：
  "如果你要重新实现一个 Claude Code，核心架构怎么设计？"
  → 单线程主循环 + h2A 双缓冲 + 上下文压缩 + 分层 SubAgent
  → 工具优先，架构从简
  → 重点在上下文管理和工具设计，不在复杂的 Agent 编排
"""


if __name__ == "__main__":

    print("╔══════════════════════════════════════════════════════╗")
    print("║  第8章：Claude Code 架构深度剖析                      ║")
    print("║  nO主循环 · h2A Steering · Compaction · SubAgent     ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n▶ 8.1 4层架构总览")
    layers = [
        "Layer 1: 用户交互层 → CLI / VS Code / Web UI / Agent SDK",
        "Layer 2: Agent 核心调度层 → nO + h2A + StreamGen + Compressor",
        "Layer 3: 工具执行与管理层 → Bash/Grep/Edit + Task/SubAgent + MCP",
        "Layer 4: 存储与持久化层 → TODO列表 + Markdown + Git",
    ]
    for l in layers:
        print(f"  {l}")

    print("\n▶ 8.3 h2A 实时 Steering 演示")
    # 异步演示（简化，实际需 asyncio 环境）
    queue = SimulatedSteeringQueue()
    print("  创建 h2A 队列完成（演示异步双缓冲概念）")
    print("  当 Agent 在等待 LLM 响应时，队列可接受注入消息")
    print("  Agent 每步结束后检查并消费注入消息")

    print("\n▶ 8.4 Context Compaction 演示")
    simulate_context_compaction()

    print("\n▶ 8.5 SubAgent 架构对比")
    print("-" * 50)
    print("Claude Code SubAgent: Task工具动态创建，无状态，完全隔离")
    print("crewAI Agent: 预定义角色，有状态，多次交互")
    print("LangGraph Agent: 图节点定义，共享State，灵活编排")
    print()
    print("选择建议：")
    print("  搜索/分析子任务 → Claude Code 风格（单次、无状态）")
    print("  内容创作协作   → crewAI 风格（角色扮演、多轮交互）")
    print("  复杂业务流程   → LangGraph 风格（状态机、条件分支）")

    print("\n▶ 8.7 设计原则速记")
    principles = [
        "Simple is Better → 单线程 + 平面消息 + Markdown",
        "Tool-First → 好工具胜过好 prompt",
        "Controllable Autonomy → 人在回路 + 权限分级",
        "Context Engineering → 92% 压缩 + Markdown 长期记忆",
    ]
    for p in principles:
        print(f"  🔑 {p}")

    print("\n✅ 第8章完成！")
