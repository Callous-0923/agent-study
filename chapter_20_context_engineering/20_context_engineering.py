"""
第20章：Context Engineering（上下文工程）—— Agent 时代的新工程学科
====================================================================

内容核对：2026-08-01
说明：标注为模拟的实现与数值用于讲解概念，不代表真实 SDK、协议或基准结果。

📌 本章目标：
  1. 理解 Context Engineering 和 Prompt Engineering 的本质区别
  2. 理解长上下文退化是任务、模型与位置相关的经验现象
  3. 学会系统化的上下文预算管理
  4. 掌握 XML 标签结构化 Prompt 的方法
  5. 理解 Anthropic 的 「Skill.md」 模式

📌 面试高频点：
  - 「Context Engineering 和 Prompt Engineering 有什么区别？」
  - 「为什么上下文越长，LLM 表现越差？」
  - 「如何管理 Agent 的上下文预算？」
  - 「System Prompt 怎么结构化设计？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心来源：Anthropic Engineering Blog (2025.09)
  "Effective context engineering for AI agents"
  Anthropic 官方 Prompt Engineering 文档
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


20.1 Prompt Engineering → Context Engineering
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prompt Engineering（过去）：
  关注点：写一个好的系统提示词
  范围：单次推理的指令文本

Context Engineering（现在）：
  关注点：管理进入 LLM 上下文窗口的所有 token
  范围：System Prompt + Tools + MCP + 对话历史 + 外部数据 + ...

类比：
  Prompt Engineering = 写好一封邮件
  Context Engineering = 管理一个人的整个信息流
    （收件箱、日历、笔记、任务列表、知识库...）

为什么需要这个转变？

  Agent 不是「单次回答问题」，而是「多轮推理循环」
  每一轮都产生新的信息（工具返回、中间结果）
  如果不管理，上下文会快速膨胀 + 质量下降

  Andrej Karpathy (2025.06)：
  "The hottest new programming language is English...
   and the hottest new engineering discipline is context engineering."


20.2 Context Rot（上下文衰减）—— 核心概念
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题本质：
  随着上下文 token 数增加，LLM 提取信息的能力逐渐衰减。

为什么会出现长上下文退化？

  1. 计算与选择难度
     标准全注意力计算随长度呈二次增长，但这解释计算成本，不等同于证明
     “注意力被稀释”。工程上还存在稀疏/分块注意力和服务端优化。

  2. 训练数据分布
     LLM 训练数据中，短文本远多于长文本
     模型对长距离依赖的经验不足

  3. 位置编码限制
     RoPE 等位置编码在长上下文时需要插值扩展
     扩展后的位置信息不如原本精确

评测方法：
  - 分别把同一关键信息放在开头、中间和结尾，测试位置敏感性
  - 逐步增加无关内容、相似干扰项和工具输出，而不只测“针在草堆”
  - 同时记录答案正确率、引用命中率、延迟与 token
  - 结果必须绑定模型快照；不存在跨模型通用的固定衰减曲线

关键启示：
  「上下文是一种边际收益递减的稀缺资源」
  → 每个 token 都在消耗 LLM 的「注意力预算」
  → 必须精打细算地使用


20.3 上下文预算管理 —— 系统化方法
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

把上下文当作「预算」来管理：

  ┌──────────────────────────────────────────┐
  │              上下文预算 = 200K tokens      │
  │                                            │
  │  ┌──────────────────┐  15% = 30K tokens   │
  │  │  System Prompt   │                      │
  │  ├──────────────────┤  5% = 10K tokens    │
  │  │  Tool Definitions│                      │
  │  ├──────────────────┤  10% = 20K tokens   │
  │  │  External Context │  (MCP/RAG 检索结果)  │
  │  ├──────────────────┤  50% = 100K tokens  │
  │  │  对话历史         │                      │
  │  ├──────────────────┤  15% = 30K tokens   │
  │  │  预留 (生成缓冲)  │                      │
  │  └──────────────────┘  5% = 10K tokens    │
  │  │  安全余量         │                      │
  │  └──────────────────┘                      │
  └──────────────────────────────────────────┘

Context Compaction 策略（面试重点！）：

  策略 1: 滑动窗口
    只保留最近 N 条消息
    优点：简单  | 缺点：丢失早期关键信息

  策略 2: 摘要压缩
    对早期对话用 LLM 生成摘要，替换原消息
    优点：保留关键信息  | 缺点：多一次 LLM 调用

  策略 3: 分层记忆
    最新 → 完整保留
    中期 → 摘要压缩
    远期 → 向量检索
    优点：平衡  | 缺点：实现复杂

  策略 4: 预算触发压缩
    优点：自动化  | 缺点：阈值需为响应和工具结果预留空间并通过回归评测
"""
import json
import time
import re
from typing import Optional
from dataclasses import dataclass
from collections import OrderedDict


class ContextBudget:
    """上下文预算管理器。

    帮助开发者系统化管理 LLM 的上下文窗口分配。
    """

    def __init__(self, total_tokens: int = 200000):
        """
        Args:
            total_tokens: 模型的上下文窗口大小。
        """
        self.total = total_tokens
        self.allocations = OrderedDict()  # {section: (tokens, data)}
        self.reserve_pct = 0.20  # 20% 预留（生成 + 安全余量）

    def allocate(self, section: str, data: str,
                 max_pct: float = None) -> int:
        """分配一段上下文给指定部分。

        Args:
            section: 部分名（如 'system_prompt'）。
            data: 内容。
            max_pct: 最大占比。

        Returns:
            实际使用的 token 估算值。
        """
        tokens = _estimate_tokens(data)
        available = self.available()

        if max_pct is not None:
            cap = int(self.total * max_pct)
            if tokens > cap:
                tokens = cap
                data = data[:cap * 3]  # 粗略截断

        if tokens > available:
            tokens = available
            data = data[:tokens * 3]

        self.allocations[section] = (tokens, data)
        return tokens

    def used(self) -> int:
        """已使用的 token 数。"""
        return sum(alloc[0] for alloc in self.allocations.values())

    def available(self) -> int:
        """可用的 token 数（扣除预留）。"""
        usable = int(self.total * (1 - self.reserve_pct))
        return usable - self.used()

    def usage_report(self) -> str:
        """生成上下文使用报告。"""
        lines = [
            f"上下文预算: {self.total:,} tokens",
            f"预留 (生成+安全): {int(self.total * self.reserve_pct):,} tokens",
            f"已使用: {self.used():,} tokens ({self.used() / self.total:.1%})",
            f"可用: {self.available():,} tokens",
            "-" * 40,
        ]
        for section, (tokens, _) in self.allocations.items():
            pct = tokens / self.total
            bar = "█" * int(pct * 50) + "░" * (50 - int(pct * 50))
            lines.append(f"  {section:20s} {tokens:>8,} tokens  {bar}")
        return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（约 4 字符 = 1 token）。"""
    return len(text) // 4


"""
20.4 结构化 Prompt —— XML 是一种选择
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

什么时候用 XML 标签？
  1. LLM 训练数据中大量存在 XML/HTML
  2. 标签天然提供语义边界
  3. 可以嵌套（层次化组织信息）

一种可读的 Prompt 结构：

  <system>
    <background_information>
      当前的上下文背景...
    </background_information>
    <instructions>
      你需要完成的任务...
    </instructions>
    <tool_guidance>
      工具使用说明...
    </tool_guidance>
    <output_description>
      输出格式要求...
    </output_description>
  </system>

结构化的好处（面试重点！）：
  1. 减少「语义混杂」：LLM 不会混淆指令和数据
  2. 防御 Prompt Injection：用户内容在 <user_input> 中，
     不会混入指令区
  3. 便于动态拼接上下文：可以程序化增删 sections
  4. 便于调试：按 tag 搜索对应的上下文块
"""


class StructuredPrompt:
    """XML 结构化 Prompt 构建器。

    展示一种 Prompt 组织方式；JSON、Markdown 标题或 provider 原生字段也可用。
    """

    def __init__(self):
        self.sections = OrderedDict()

    def add_section(self, tag: str, content: str):
        """添加一个结构化 section。

        Args:
            tag: XML 标签名。
            content: section 内容。
        """
        self.sections[tag] = content

    def build(self) -> str:
        """构建完整的结构化 Prompt。

        Returns:
            完整的 XML 标记文本。
        """
        parts = ["<system>"]
        for tag, content in self.sections.items():
            parts.append(f"  <{tag}>")
            parts.append(f"    {content}")
            parts.append(f"  </{tag}>")
        parts.append("</system>")
        return "\n".join(parts)

    @staticmethod
    def extract_section(prompt: str, tag: str) -> Optional[str]:
        """从结构化 Prompt 中提取指定 section。

        Args:
            prompt: 完整的 Prompt 文本。
            tag: 要提取的标签名。

        Returns:
            section 内容，未找到返回 None。
        """
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, prompt, re.DOTALL)
        return match.group(1).strip() if match else None


def demo_structured_prompt():
    """演示结构化 Prompt 的构建与使用。"""
    print("=" * 60)
    print("  XML 结构化 Prompt 演示")
    print("=" * 60)

    builder = StructuredPrompt()
    builder.add_section(
        "background",
        "用户是一名 Python 开发者，正在学习 AI Agent 技术。"
    )
    builder.add_section(
        "instructions",
        "请用简洁的语言解释上下文工程的核心概念。"
        "不要超过 200 字。使用中文回答。"
    )
    builder.add_section(
        "tool_guidance",
        "如果用户问代码相关的问题，使用 code_exec 工具。"
        "如果用户问事实性问题，使用 search 工具。"
    )
    builder.add_section(
        "output_format",
        "用 Markdown 格式输出，包含标题和要点列表。"
    )
    builder.add_section(
        "user_input",
        "什么是 Context Rot？为什么它很重要？"
    )

    full_prompt = builder.build()
    print(full_prompt)

    # 演示动态修改
    print("\n  📝 提取 user_input section:")
    user_input = StructuredPrompt.extract_section(
        full_prompt, "user_input"
    )
    print(f"  → {user_input}")


"""
20.5 Prompt 的「上下文衰减曲线」与 Golden Zone
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

通用实践经验：

  Prompt 太简单（硬编码 if-else 级别）：
    → 过于脆弱，不能处理变体

  Prompt 太复杂/笼统（万言书级别）：
    → LLM 抓不住重点
    → 容易出现「上下文衰减」

  Golden Zone：
    → 信息密度最高、恰好覆盖所需、不冗余
    → 每段内容要有明确的「信息价值 vs token 成本」

20.6 Anthropic Skill.md 模式
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Skill 文件是一种按需加载过程知识的上下文工程实践：

  每个 Skill 是一个 Markdown 文件（skill.md）：
  ```
  # Skill: 代码审查
  ## 目的
  审查代码的 Bug、安全漏洞和风格问题

  ## 触发条件
  - 用户说 "review" / "审查" / "check"
  - 用户提供了代码片段或文件路径

  ## 执行步骤
  1. 读取代码
  2. 检查常见问题（空指针、注入、性能瓶颈）
  3. 生成审查报告

  ## 输出格式
  - Bug (危险等级: 高/中/低)
  - 建议的修复方案
  - 代码风格建议
  ```

  优势：
    1. Markdown 对 LLM 友好（训练数据中大量存在）
    2. 人类可维护（非技术人员也能写）
    3. 结构清晰（LLM 容易遵循）
    4. 可与工具、MCP Server 或本地运行时组合

  文件名、发现规则和可执行能力取决于具体运行时，不是跨产品统一协议。


20.7 上下文工程的实践 Checklist
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 结构层面
  ☐ 使用 XML 标签明确分隔不同部分
  ☐ 用户输入与可信指令分区；标签提升可读性，但不能防止提示注入
  ☐ 工具定义用独立的 tool_guidance section

✅ 内容层面
  ☐ 每条信息都有明确的「为何需要它」
  ☐ 删除冗余的、无信息量的内容
  ☐ 优先保留高频使用的信息

✅ 动态层面
  ☐ 设置上下文预算上限
  ☐ 实现自动压缩策略（摘要/滑动窗口）
  ☐ 监控上下文使用率（类似内存监控）

✅ 编码 Agent 经验（来自 Ch8）
  ☐ 压缩阈值按上下文预算和回归评测确定
  ☐ 外部记忆可审计、可删除、可做权限控制
  ☐ TODO 列表管理短期任务状态
"""


def demo_context_budget():
    """演示上下文预算管理。"""
    print("=" * 60)
    print("  上下文预算管理演示")
    print("=" * 60)

    budget = ContextBudget(200000)

    # 分配各部分
    system_prompt = "你是一个 AI 助手..." * 100
    budget.allocate("system_prompt", system_prompt, max_pct=0.15)
    print(f"  system_prompt: {budget.allocations['system_prompt'][0]:,} tokens")

    tool_defs = "工具定义: search, calculator, ..." * 50
    budget.allocate("tool_definitions", tool_defs, max_pct=0.05)
    print(f"  tool_definitions: {budget.allocations['tool_definitions'][0]:,} tokens")

    external = "检索结果: ..." * 200
    budget.allocate("external_context", external, max_pct=0.15)
    print(f"  external_context: {budget.allocations['external_context'][0]:,} tokens")

    history = "对话历史..." * 500
    budget.allocate("conversation_history", history)
    print(f"  conversation_history: {budget.allocations['conversation_history'][0]:,} tokens")

    print("\n" + budget.usage_report())


"""
20.8 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. Context Engineering = 管理进入 LLM 的所有 token
   - 从「写 prompt」升级到「管理信息流」
   - 上下文 = 稀缺资源（边际收益递减）

2. Context Rot
   - 长上下文导致注意力稀释（n² 关系）
   - 训练数据分布偏向短文本
   - 位置编码扩展损失精度

3. 上下文预算管理
   - 类似内存管理：分配 → 监控 → 压缩
   - 策略：滑动窗口 / 摘要压缩 / 分层记忆 / 预算触发

4. 结构化 Prompt（XML）
   - 明确边界：用户输入隔离有助于理解，但不是安全边界
   - 便于拼接：程序化增删
   - 提高可读性：LLM 和人类都受益

面试速记：
  "什么是 Context Engineering？"
  → 从写 Prompt 升级到管理所有进 LLM 的 token
  → 核心挑战：Context Rot（注意力随长度衰减）
  → 解决方案：预算管理 + 结构化 + 压缩策略
  → 结构格式应与模型、SDK 和评测匹配；XML 不是唯一最佳格式
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第20章：Context Engineering（上下文工程）              ║")
    print("║  Context Rot · 预算管理 · XML Prompt · Skill.md      ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n▶ 20.2 Context Rot 评测方法")
    print("-" * 50)
    print("  改变信息位置、上下文长度和干扰项，分别测量")
    print("  记录正确率、引用命中率、延迟与 token，并绑定模型版本")

    demo_context_budget()
    demo_structured_prompt()

    print("\n▶ Context Engineering Checklist")
    print("-" * 50)
    items = [
        "使用 XML 标签分隔 Prompt 各部分",
        "用户输入放在独立标签中（防注入）",
        "设置上下文预算上限，监控使用率",
        "实现自动压缩（摘要/滑动窗口/分层记忆）",
        "每条信息都要有「信息价值 vs token 成本」的权衡",
        "Markdown 文件做长期记忆（Claude Code 经验）",
    ]
    for item in items:
        print(f"  ✅ {item}")

    print("\n✅ 第20章完成！")
