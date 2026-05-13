"""
第16章：MemGPT / Letta —— Agent 记忆的「操作系统」
=====================================================

📌 本章目标：
  1. 理解 MemGPT 的核心思想：「把 LLM 当作操作系统来管」
  2. 掌握内存层次结构：Core Memory ↔ Recall Memory ↔ Archival Memory
  3. 理解 Heartbeat 机制（自我纠错）
  4. 理解 Sleep-Time Compute（后台记忆整理）
  5. 了解 Letta Filesystem（2025年最新发现：文件系统做记忆效果惊人）

📌 面试高频点：
  - MemGPT 怎么解决 LLM 的上下文窗口限制？
  - Core Memory 和 Archival Memory 的区别？
  - Heartbeat 是什么？为什么需要它？
  - Letta 的「文件系统做记忆」为什么效果好？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
论文: "MemGPT: Towards LLMs as Operating Systems" (2023)
项目: github.com/letta-ai/letta (原 MemGPT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


16.1 核心思想：把 LLM 当作操作系统
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

类比：计算机操作系统管理内存的方式

  操作系统内存层次:
    ┌──────────┐  最快/最小
    │  寄存器   │
    ├──────────┤
    │  L1/L2  │
    │  Cache   │
    ├──────────┤
    │   RAM    │  ← 主内存
    ├──────────┤
    │  磁盘     │  ← 持久化
    └──────────┘  最慢/最大

  MemGPT 的内存层次:
    ┌──────────────┐  最快/在上下文中
    │   Core Memory  │  ← 当前对话 + 记忆块
    │   (上下文内)    │     (类似 RAM)
    ├──────────────┤
    │  Recall Memory │  ← 完整对话历史
    │  (上下文外)     │     (磁盘，可检索)
    ├──────────────┤
    │ Archival Memory│  ← 外部知识库
    │   (向量数据库)  │     (归档存储)
    └──────────────┘  最慢/最大

核心洞察：
  - LLM 的上下文窗口 = 操作系统的 RAM
  - 上下文有限，但通过「换页」技术可以管理无限容量的信息
  - OS 用虚拟内存换页 → MemGPT 用记忆管理工具换页


16.2 MemGPT 的内存管理机制
━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────────────────────────────────────────┐
│                 MemGPT Agent                         │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │            Core Memory (上下文内)              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │  │
│  │  │ System   │ │ Memory   │ │ Conversation │ │  │
│  │  │ Prompt   │ │ Blocks   │ │ History      │ │  │
│  │  └──────────┘ └──────────┘ └──────────────┘ │  │
│  └──────────────────────────────────────────────┘  │
│                       │                             │
│            ┌──────────┴──────────┐                  │
│            │   对话历史管理工具     │                  │
│            │  ─────────────────  │                  │
│            │  conversation_search │ ← 搜索完整历史    │
│            │  core_memory_append  │ ← 编辑记忆块      │
│            │  core_memory_replace │ ← 替换记忆块      │
│            └─────────────────────┘                  │
│                       │                             │
│  ┌────────────────────┴──────────────────────────┐ │
│  │              外部记忆（上下文外）                │ │
│  │  ┌────────────────┐  ┌──────────────────────┐ │ │
│  │  │ Recall Memory   │  │  Archival Memory      │ │ │
│  │  │ (对话历史全文)    │  │  (向量库 + 知识检索)   │ │ │
│  │  │ archival_memory │  │  archival_memory     │ │ │
│  │  │ _search        │  │  _insert             │ │ │
│  │  └────────────────┘  └──────────────────────┘ │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘

核心工具（Agent 通过 Function Calling 调用）：
  1. send_message: 向用户发送消息（MemGPT 唯一的消息出口）
  2. core_memory_append: 向 Core Memory 块追加信息
  3. core_memory_replace: 替换 Core Memory 块的内容
  4. conversation_search: 搜索完整对话历史
  5. archival_memory_insert: 将信息存入归档记忆（向量化）
  6. archival_memory_search: 从归档记忆中检索


16.3 Heartbeat 机制 —— Agent 的「自我意识」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统 Agent 的问题：
  工具调用 → 返回结果 → 给用户 → 结束
  Agent 没有机会检查「我做得对不对」

Heartbeat 的解决方案：
  每个工具的返回结果中包含一个 request_heartbeat 参数
  如果设为 true → 工具执行完毕后，控制权交还给 Agent
  Agent 可以：检查结果 → 发现错误 → 自动重试

代码示意：
  # 工具返回结果
  {
    "tool_call_id": "call_123",
    "content": "搜索结果为空",
    "request_heartbeat": true  // ← 请求心跳！
  }

  # Agent 收到心跳后
  → "搜索结果为空？我换一个关键词试试"
  → 再次调用 search("换一个搜索词")
  → 这次找到了！

这类似于：
  你在命令行执行 ls 之后，shell 不会直接退出
  而是等你输入下一个命令
  Heartbeat 就是给 Agent 的「命令提示符」


16.4 Sleep-Time Compute —— 后台记忆整理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MemGPT v2 的创新（2025年）。

核心思想：
  Agent 在「休眠」时（用户不互动时），
  启动一个后台 Agent 来整理记忆。

做什么？
  1. 浏览最近的对话历史
  2. 提取重要信息 → 写入 Core Memory
  3. 删除过时的信息
  4. 更新用户画像
  5. 合并重复的记忆

类比：
  就像你睡觉时，大脑在整理白天的记忆
  不重要的事情遗忘，重要的事情巩固

效果：
  Letta 在 LoCoMo 基准测试上达到 74.0% 准确率
  （仅用文件系统 + GPT-4o-mini）


16.5 Letta Filesystem ——「文件系统 = 最好的记忆工具」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2025年8月，Letta 发表了一篇震撼论文：
  "Is a Filesystem All You Need?"
  → LoCoMo 基准测试 74.0%（超过专用记忆工具 Mem0 的 68.5%）

为什么文件系统做记忆效果惊人？

  1. 文件操作是 LLM 最熟悉的范式
     - grep / open / close / search_files
     - 这些工具在 LLM 训练数据中大量出现
     - Agent 对它们的使用非常自然

  2. 简单即正确
     - 专用记忆工具引入额外复杂性
     - 多了出错的环节
     - Agent 需要理解专用工具的特殊语义

  3. 无需学习新范式
     - 每个记忆工具（Mem0/Zep/LangMem）都有自己的 API
     - LLM 需要「学会」每个新工具
     - 而文件操作是 LLM 的「母语」

核心启示（面试金句！）：
  "Agent 的记忆质量取决于 Agent 管理上下文的能力，
   而不是记忆工具的复杂度。文件系统 + 好用工具 = 最佳记忆方案。"


16.6 MemGPT 架构的简化实现
━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import Optional
import json
import time
import hashlib


class MemGPTAgent:
    """MemGPT 架构的简化实现 —— 展示核心概念。

    实现：
      1. Core Memory 块（System + Human + Persona）
      2. 记忆编辑工具（core_memory_append/replace）
      3. Conversation Search（对话历史搜索）
      4. 上下文压缩（对话历史过长时自动压缩）
    """

    def __init__(self, system_prompt: str = ""):
        """初始化 MemGPT Agent。

        Args:
            system_prompt: 系统提示词。
        """
        # Core Memory 块
        self.core_memory = {
            "human": "用户偏好和背景信息",
            "persona": "AI助手，专业且友好",
            "system": system_prompt or "你是一个有用的 AI 助手。",
        }

        # 完整对话历史（Recall Memory）
        self.conversation = []

        # 归档记忆（简化版：字典存储）
        self.archival = {}

        # 已压缩的摘要
        self.compacted_summary = ""

    def _build_context(self, max_recent: int = 10) -> list[dict]:
        """构建发给 LLM 的上下文。

        包含：
          1. Core Memory 块
          2. 压缩摘要（如果有）
          3. 最近 N 条对话

        Args:
            max_recent: 最大保留的最近消息数。

        Returns:
            上下文消息列表。
        """
        context = []

        # Core Memory 组装
        core_text = "\n".join(
            f"### {key}\n{value}" for key, value in self.core_memory.items()
        )
        context.append({
            "role": "system",
            "content": f"<memory>\n{core_text}\n</memory>",
        })

        if self.compacted_summary:
            context.append({
                "role": "system",
                "content": f"<summary>\n{self.compacted_summary}\n</summary>",
            })

        # 最近 N 条对话
        recent = self.conversation[-max_recent:] if self.conversation else []
        context.extend(recent)

        return context

    def core_memory_append(self, block: str, content: str):
        """向 Core Memory 块追加内容。

        Args:
            block: 块名（human/persona）。
            content: 追加的内容。
        """
        if block in self.core_memory:
            self.core_memory[block] += "\n" + content

    def core_memory_replace(self, block: str, content: str):
        """替换 Core Memory 块的内容。

        Args:
            block: 块名。
            content: 新内容。
        """
        if block in self.core_memory:
            self.core_memory[block] = content

    def archival_memory_insert(self, key: str, content: str):
        """向归档记忆插入一条信息。

        Args:
            key: 检索键。
            content: 信息内容。
        """
        self.archival[key] = {
            "content": content,
            "timestamp": time.time(),
        }

    def archival_memory_search(self, query: str) -> str:
        """从归档记忆搜索（简化版关键词匹配）。

        Args:
            query: 搜索词。

        Returns:
            匹配的结果。
        """
        results = []
        for key, entry in self.archival.items():
            if query.lower() in key.lower() or query.lower() in entry["content"].lower():
                results.append(f"[{key}] {entry['content']}")
        return "\n".join(results[-5:]) if results else "未找到相关信息"

    def compact_conversation(self):
        """对话压缩：早期对话 → 摘要，放入 compacted_summary。

        这是 MemGPT 最核心的操作之一：
          当对话历史过长时，对早期部分做摘要压缩。
        """
        if len(self.conversation) <= 20:
            return

        # 取最早的 10 条对话做压缩
        old_part = self.conversation[:10]
        self.conversation = self.conversation[10:]

        # 简化的摘要生成
        topics = set()
        for msg in old_part:
            for word in msg.get("content", "").split():
                if len(word) > 3:
                    topics.add(word.lower())

        new_summary = (
            f"此前讨论了以下话题：{', '.join(list(topics)[:10])}..."
        )

        if self.compacted_summary:
            self.compacted_summary += "\n" + new_summary
        else:
            self.compacted_summary = new_summary

    def heartbeat_should_continue(self, last_tool_result: str) -> bool:
        """判断是否需要 Heartbeat（继续给 Agent 控制权）。

        如果工具调用失败或者返回不充分，
        返回 True 让 Agent 继续思考和行动。

        Args:
            last_tool_result: 上一步工具调用的结果。

        Returns:
            是否需要继续。
        """
        # 模拟判断：如果结果包含错误信号
        error_signals = ["未找到", "错误", "失败", "抱歉", "无法"]
        return any(sig in last_tool_result for sig in error_signals)


def demo_memgpt_workflow():
    """演示 MemGPT 的完整工作流。"""
    print("=" * 60)
    print("  MemGPT 记忆架构演示")
    print("=" * 60)

    agent = MemGPTAgent()

    # Step 1: 展示初始 Core Memory
    print("\n  🧠 初始 Core Memory:")
    for key, value in agent.core_memory.items():
        print(f"    [{key}] {value}")

    # Step 2: Agent 学习用户信息 → 写入 Core Memory
    print("\n  📝 用户告诉 Agent 自己的偏好")
    agent.core_memory_append("human", "喜欢简洁的回答风格")
    agent.core_memory_append("human", "是Python开发者，经常需要代码帮助")
    agent.core_memory_replace("persona", "Python 专家助手，回答简洁专业")

    print("    更新后的 Core Memory:")
    for key, value in agent.core_memory.items():
        print(f"    [{key}] {value}")

    # Step 3: 模拟对话
    print("\n  💬 模拟对话")
    dialogs = [
        ("user", "帮我写一个快速排序算法"),
        ("assistant", "好的，这是一个 Python 快速排序实现：\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)"),
        ("user", "谢谢！再帮我分析一下时间复杂度"),
        ("assistant", "快速排序的平均时间复杂度是 O(n log n)，最坏情况是 O(n²)..."),
    ]
    for role, content in dialogs:
        agent.conversation.append({"role": role, "content": content})

    print(f"    当前对话数: {len(agent.conversation)}")

    # Step 4: 写入归档记忆
    print("\n  📦 归档记忆")
    agent.archival_memory_insert(
        "quicksort", "快速排序：O(n log n) 平均，O(n²) 最坏，非稳定排序"
    )
    agent.archival_memory_insert(
        "python_best_practice", "Python编码规范：PEP 8, 类型注解, 单元测试"
    )
    agent.archival_memory_insert(
        "user_preference", "用户偏好简洁代码，不喜欢过度注释"
    )
    print(f"    已存入 {len(agent.archival)} 条归档记忆")

    # Step 5: 搜索归档记忆
    print("\n  🔍 搜索归档记忆")
    result = agent.archival_memory_search("排序")
    print(f"    搜索「排序」→ {result}")

    # Step 6: 模拟对话压缩
    print("\n  🗜️ 对话压缩演示")
    # 模拟大量对话
    for i in range(50):
        agent.conversation.append({
            "role": "user",
            "content": f"问题{i}：这是一条测试消息",
        })
        agent.conversation.append({
            "role": "assistant",
            "content": f"回答{i}：这是一条回复",
        })

    print(f"    压缩前: {len(agent.conversation)} 条消息")
    agent.compact_conversation()
    print(f"    压缩后: {len(agent.conversation)} 条消息")
    print(f"    摘要: {agent.compacted_summary[:100]}...")

    # Step 7: Heartbeat 机制演示
    print("\n  💓 Heartbeat 机制演示")
    test_cases = [
        ("查询成功：找到 3 条结果", False),
        ("未找到相关数据，请更换搜索词", True),
        ("工具调用失败：API 超时", True),
        ("计算完成：结果为 42", False),
    ]
    for result, expected_heartbeat in test_cases:
        needs_heartbeat = agent.heartbeat_should_continue(result)
        status = "✅" if needs_heartbeat == expected_heartbeat else "❌"
        print(f"    {status} 「{result[:25]}...」→ 需要心跳: {needs_heartbeat}")

    # Step 8: 展示 LLM 上下文
    print("\n  📋 发给 LLM 的上下文结构")
    context = agent._build_context(max_recent=3)
    for i, msg in enumerate(context):
        role = msg["role"]
        content = msg["content"][:80].replace("\n", " ")
        print(f"    [{i}] {role}: {content}...")


"""
16.7 本章总结
━━━━━━━━━━━━━

核心要点回顾：

1. MemGPT = 把 LLM 当操作系统管理
   - Core Memory = RAM（上下文中，快速访问）
   - Recall Memory = 磁盘（对话历史全文，可检索）
   - Archival Memory = 归档（向量库 + 外部知识）

2. Heartbeat 机制
   - 工具调用后把控制权交还给 Agent
   - Agent 可以检查结果、发现错误、自动重试
   - 类似命令行的「命令提示符」

3. Sleep-Time Compute (MemGPT v2)
   - Agent 休眠时后台整理记忆
   - 提取重要信息、删除冗余、更新画像

4. Letta Filesystem 的启示
   - 文件系统做记忆 = 简单 + 效果好
   - LoCoMo 74.0%（超过专用工具的 68.5%）
   - Agent 不需要复杂的记忆工具，需要好用的工具

面试速记：
  "MemGPT 的核心创新是什么？"
  → OS 级内存管理：Core Memory + Recall + Archival
  → 通过工具让 Agent 自己管理自己的记忆
  → Heartbeat 机制实现自我纠错
  → 「文件系统就是最好的记忆工具」
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第16章：MemGPT / Letta 记忆架构                        ║")
    print("║  Core Memory · Heartbeat · Sleep-Time · Filesystem   ║")
    print("╚══════════════════════════════════════════════════════╝")

    demo_memgpt_workflow()

    print("\n▶ MemGPT 内存层次对应关系")
    print("-" * 50)
    layers = [
        ("Core Memory", "RAM", "当前上下文 + 可编辑记忆块"),
        ("Recall Memory", "磁盘", "完整对话历史，可搜索"),
        ("Archival Memory", "归档", "向量数据库，外部知识"),
    ]
    for mem, os, desc in layers:
        print(f"  {mem:18s} ↔ {os:6s} → {desc}")

    print("\n✅ 第16章完成！")
