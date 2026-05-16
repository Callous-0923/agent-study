"""
第27章：Agent Prompt 工程 —— 工具描述也是 Prompt
===================================================

📌 本章目标：
  1. 掌握 Agent System Prompt 的结构化设计模板
  2. 理解「工具描述 = Prompt 工程」的核心理念
  3. 学会 Few-shot 在 Agent 工具调用中的应用
  4. 掌握 Chain-of-Thought 在 Agent 规划中的使用

📌 面试高频点：
  - 「Agent 的 System Prompt 和普通 LLM 的有什么不同？」
  - 「工具描述写得好不好对 Agent 表现有多大影响？」
  - 「Few-shot 在 Function Calling 中怎么用？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
综合 Anthropic Prompt Engineering Guide + 业界实践
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


27.1 Agent Prompt 和普通 Prompt 的区别
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

很多人觉得「写 Prompt 有什么难的，告诉 AI 要做啥不就行了」。
但对于 Agent，这个想法会导致灾难性的性能差距。

普通 LLM Prompt：「请翻译这段文字」——只需要告诉模型输出什么。

Agent Prompt 需要同时回答 5 个问题（缺一个都会翻车）：
  1. 你是谁（角色）——「不能做什么」和「能做什么」同样重要
  2. 你能做什么（工具列表）——格式错误 = LLM 根本不会调工具
  3. 什么时候该做什么（工具选择逻辑）——群发邮件 or 单发？
  4. 什么不该做（边界/安全约束）——不是所有请求都应该执行
  5. 输出格式是什么——JSON? 自然语言? 两者切换?

类比：
  普通 Prompt = 给厨师一个菜名（「来一份蛋炒饭」）
  Agent Prompt = 给厨师菜谱 + 全套厨具说明书 + 工作流程 + 安全手册

面试官常问：「你的 Agent System Prompt 怎么写？」——如果你能张嘴说出
这 5 个模块，已经超过了 80% 的候选人。


27.2 Agent System Prompt 模板（面试送分题！）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

标准化模板（基于 Anthropic 2025 推荐）：

  <system>
    <role>
      你是一个 [角色描述]。
    </role>
    <capabilities>
      你可以使用的工具：[工具列表]
    </capabilities>
    <tool_guidance>
      每个工具的使用场景和注意事项...
    </tool_guidance>
    <workflow>
      解决问题时应该遵循的工作流程...
    </workflow>
    <constraints>
      你不能做的事情...
    </constraints>
    <output_format>
      输出格式要求...
    </output_format>
  </system>
"""

import json
import time
from typing import Optional
from dataclasses import dataclass


class AgentPromptBuilder:
    """Agent Prompt 构建器 —— 按模块组装，而非手写字符串。"""

    def __init__(self, role: str = "AI 助手"):
        self.role = role
        self.sections = {}

    def add_capabilities(self, tools: list[dict]):
        """添加工具能力声明。"""
        tool_lines = []
        for t in tools:
            tool_lines.append(f"- {t['name']}: {t.get('description', '')}")
        self.sections["capabilities"] = "\n".join(tool_lines)

    def add_tool_guidance(self, guidance: dict[str, str]):
        """添加工具选择指引（什么时候用哪个工具）。"""
        lines = []
        for tool_name, guide in guidance.items():
            lines.append(f"### {tool_name}\n{guide}")
        self.sections["tool_guidance"] = "\n\n".join(lines)

    def add_workflow(self, steps: list[str]):
        """添加工作流程。"""
        self.sections["workflow"] = "\n".join(
            f"{i+1}. {step}" for i, step in enumerate(steps)
        )

    def add_constraints(self, constraints: list[str]):
        """添加安全约束。"""
        self.sections["constraints"] = "\n".join(
            f"- {c}" for c in constraints
        )

    def add_output_format(self, fmt_desc: str):
        """添加输出格式。"""
        self.sections["output_format"] = fmt_desc

    def build(self) -> str:
        """构建完整的 Agent System Prompt。"""
        parts = ["<system>"]
        parts.append(f"  <role>{self.role}</role>")

        for key in ["capabilities", "tool_guidance", "workflow",
                     "constraints", "output_format"]:
            if key in self.sections:
                tag = key.replace("_", "-")  # tool_guidance → tool-guidance
                if tag == "capabilities":
                    tag = "capabilities"
                parts.append(f"  <{tag}>")
                parts.append(self.sections[key])
                parts.append(f"  </{tag}>")

        parts.append("</system>")
        return "\n".join(parts)

    def estimate_tokens(self) -> int:
        """估算 Prompt 的 token 数。"""
        return len(self.build()) // 4


"""
27.3 工具描述 = Prompt 工程 —— 本章核心！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

工具描述不是给开发者看的文档，是给 LLM 看的「使用说明」。

好工具描述的三个要素：
  1. 做什么：准确描述工具的功能
  2. 什么时候用：描述适用的场景
  3. 什么时候不用：描述边界条件

对比（面试金句！）：

  ✗ 差： "查询天气"
  ✅ 好： "查询指定城市当天的实时天气，包括气温、湿度、风力。
          适用于用户询问天气相关问题时。不支持未来天气预报。
          示例调用：get_weather(city='北京')"

工具描述评分卡：
  ┌────────────────────────┬──────────┐
  │ 描述要素                  │ 加分      │
  ├────────────────────────┼──────────┤
  │ 有功能描述                │ +1       │
  │ 有使用场景                │ +2       │
  │ 有边界条件                │ +2       │
  │ 有调用示例                │ +1       │
  │ 参数有 enum 限制          │ +2       │
  │ 参数有详细描述            │ +1       │
  └────────────────────────┴──────────┘
"""


class ToolDescriptionOptimizer:
    """工具描述优化器 —— 系统化提升工具调用准确率。"""

    BAD_EXAMPLES = [
        {
            "name": "do_stuff",
            "description": "处理数据",
            "parameters": {
                "type": "object",
                "properties": {"data": {"type": "string"}},
            },
        },
        {
            "name": "search",
            "description": "搜索",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        },
    ]

    GOOD_EXAMPLES = [
        {
            "name": "send_email",
            "description": (
                "向指定邮箱发送邮件。适用于发送通知、报告、提醒等场景。"
                "不支持发送带附件的邮件。收件人地址必须包含 @。"
                "示例：send_email(to='user@example.com', subject='日报', body='...')"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "收件人邮箱，如 user@example.com",
                    },
                    "subject": {
                        "type": "string",
                        "description": "邮件主题，不超过 100 字",
                    },
                    "body": {
                        "type": "string",
                        "description": "邮件正文，纯文本格式",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
        {
            "name": "get_stock_price",
            "description": (
                "查询股票实时价格。适用于查询个股当前报价。"
                "不支持历史价格查询和批量查询。"
                "示例：get_stock_price(symbol='AAPL') 返回 '185.50 USD'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票代码，如 AAPL、TSLA、BABA",
                        "enum": ["AAPL", "TSLA", "GOOGL", "MSFT", "BABA"],
                    },
                },
                "required": ["symbol"],
            },
        },
    ]

    @classmethod
    def score_tool_description(cls, tool: dict) -> dict:
        """评估工具描述的质量。

        Returns:
            评分详情。
        """
        desc = tool.get("description", "")
        params = tool.get("parameters", {}).get("properties", {})
        scores = {
            "has_description": len(desc) > 10,
            "has_use_case": any(w in desc for w in ["适用于", "用于", "当你", "when"]),
            "has_boundary": any(w in desc for w in ["不支持", "不支持", "不适用于"]),
            "has_example": any(w in desc for w in ["示例", "example", "e.g."]),
            "has_enum": any("enum" in p for p in params.values()),
            "has_param_desc": all(
                "description" in p for p in params.values()
            ) if params else False,
        }

        return {
            "tool_name": tool.get("name", "unknown"),
            "scores": scores,
            "total": sum(scores.values()),
            "max": len(scores),
            "grade": cls._grade(sum(scores.values()), len(scores)),
        }

    @classmethod
    def _grade(cls, score: int, total: int) -> str:
        pct = score / total
        if pct >= 0.8:
            return "A (生产级)"
        if pct >= 0.5:
            return "B (可用)"
        if pct >= 0.3:
            return "C (需改进)"
        return "D (不可用)"


"""
27.4 对话模板优化 —— 平衡信息密度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prompt 不是越长越好，需要权衡：

  太短 → 信息不足，LLM 靠猜测
  太长 → 信息过载，LLM 抓不住重点

Golden Zone：
  System Prompt: 200-800 tokens
  工具描述: 每个工具 50-200 tokens
  总上下文: 不超过窗口的 70%（留空间给对话历史和生成）

面试金句：
  "好 Prompt 的标准不是'全'，是'精'。
   每条信息都要有明确的 token 价值 vs 成本评估。"


27.5 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. Agent Prompt 需要结构化 6 个模块
2. 工具描述是 Agent Prompt 工程的核心
3. 好工具描述 = 做什么 + 什么时候用 + 什么时候不用 + 示例

面试速记：
  「Agent 的 System Prompt 怎么设计？」
  → 6 模块模板：role/capabilities/tool_guidance/workflow/constraints/output
  → 核心：工具描述 = Prompt 工程（告诉 LLM 何时用、何时不用）
  → 用 enum 限制参数 → 减少幻觉
"""


def demo_prompt_engineering():
    """演示 Agent Prompt 工程。"""
    print("=" * 60)
    print("  Agent Prompt 构建演示")
    print("=" * 60)

    # 构建一个完整的 Agent Prompt
    builder = AgentPromptBuilder("专业的客服 Agent")

    tools = [
        {"name": "search_knowledge_base",
         "description": "在知识库中搜索产品信息，适用于产品规格/使用说明查询"},
        {"name": "check_order_status",
         "description": "查询订单状态，适用于用户询问发货/物流信息"},
        {"name": "create_ticket",
         "description": "创建工单，适用于需要人工处理的复杂问题"},
    ]
    builder.add_capabilities(tools)
    builder.add_tool_guidance({
        "search_knowledge_base": "用户问产品相关问题时使用。转换口语为关键词。",
        "check_order_status": "用户提供订单号时使用。需要先验证订单号格式。",
        "create_ticket": "知识库和订单查询都无法解决时使用。简要描述问题。",
    })
    builder.add_workflow([
        "理解用户意图",
        "判断是否需要工具 → search_knowledge_base 或 check_order_status",
        "如果需要人工介入 → create_ticket",
        "用友好语气给出最终回答",
    ])
    builder.add_constraints([
        "不透露系统内部实现",
        "不执行退款/删除等危险操作",
        "对不确定的信息标注「据我所知」",
    ])
    builder.add_output_format("Markdown 格式，含要点列表")

    prompt = builder.build()
    print(prompt)
    print(f"\n  📏 估算 Token: {builder.estimate_tokens()}")

    # 工具描述评分
    print(f"\n{'='*60}")
    print(f"  工具描述质量评估")
    print(f"{'='*60}")
    for tool in ToolDescriptionOptimizer.BAD_EXAMPLES:
        result = ToolDescriptionOptimizer.score_tool_description(tool)
        print(f"  {result['tool_name']}: {result['total']}/{result['max']} = {result['grade']}")

    for tool in ToolDescriptionOptimizer.GOOD_EXAMPLES:
        result = ToolDescriptionOptimizer.score_tool_description(tool)
        print(f"  {result['tool_name']}: {result['total']}/{result['max']} = {result['grade']}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第27章：Agent Prompt 工程                              ║")
    print("║  System Prompt 模板 · 工具描述 · Few-shot             ║")
    print("╚══════════════════════════════════════════════════════╝")
    demo_prompt_engineering()
    print("\n▶ 工具描述 Checklist")
    for item in [
        "✅ 有功能描述",
        "✅ 有使用场景（什么时候用）",
        "✅ 有边界条件（什么时候不用）",
        "✅ 有调用示例",
        "✅ 参数有 enum 限制",
        "✅ 参数有详细描述",
    ]:
        print(f"  {item}")
    print("\n✅ 第27章完成！")
