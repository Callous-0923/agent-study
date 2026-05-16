"""
第2章：Agent 核心组件深度解析
=====================================

📌 本章目标：
  1. 深入理解 Agent 的三大组件：规划器 / 记忆系统 / 工具调用
  2. 理解不同规划策略的优劣（ReAct vs Plan-Execute）
  3. 掌握三种记忆类型：短期 / 长期 / 工作记忆
  4. 学会设计高质量的 Agent 工具

📌 面试高频点：
  - Agent 记忆的类型和区别
  - 如何解决 Agent 的上下文窗口限制
  - 工具设计的最佳实践（description 怎么写效果好）


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.1 规划器 (Planner) —— Agent 的「指挥中心」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

规划器负责决定「下一步该做什么」。
不同的规划策略适用于不同场景：

┌──────────────────┬──────────────────────┬──────────────────────┐
│    策略          │       核心思想        │      适用场景         │
├──────────────────┼──────────────────────┼──────────────────────┤
│ ReAct           │ 边推理边行动          │ 简单-中等复杂度任务   │
│ Plan-Execute    │ 先制定整体计划再执行   │ 复杂多步骤任务        │
│ Reflexion       │ 执行后自我反思改进     │ 需要迭代优化的任务    │
│ LLMCompiler     │ 分析依赖后并行执行     │ 步骤可并行的任务      │
└──────────────────┴──────────────────────┴──────────────────────┘

面试常问：「ReAct 和 Plan-Execute 有什么区别？」
答：
  ReAct：每一步都重新推理，灵活但效率低。
  Plan-Execute：一次性规划全部步骤，按计划执行，高效但不灵活。
  实际项目中常结合使用：高层用 Plan-Execute，每步内用 ReAct。
"""

import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


def call_llm_with_json(prompt: str, system_msg: str = "") -> dict:
    """调用 LLM 并强制返回 JSON 格式。

    在 Agent 开发中，经常需要 LLM 返回结构化数据
    （如规划步骤、实体提取结果等）。

    Args:
        prompt: 用户提示词。
        system_msg: 系统消息（可选）。

    Returns:
        LLM 返回的 JSON 字典。
    """
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)


"""
2.1.1 Plan-Execute 规划器 —— 先把计划列出来再执行
为什么需要规划器？设想你让 Agent「帮我做一个市场调研报告」。
这不是一步能完成的任务——需要查资料、分析数据、写报告多个步骤。
如果每步都让 LLM 临时判断下一步做什么（纯 ReAct 模式），
容易跑偏或遗漏步骤。Plan-Execute 的思想是「先列计划，再一步步执行」，
就像项目管理中的甘特图——这也是面试中最常被问到的 Agent 设计模式之一。
"""


def plan_execute(task: str) -> dict:
    """Plan-Execute 规划器：先制定完整的执行计划。

    适用场景：
      - 任务明确，不需要动态调整
      - 多个步骤之间有依赖关系
      - 需要整体把控任务进度

    Args:
        task: 用户的任务描述。

    Returns:
        包含计划步骤的字典。
    """
    system_msg = (
        "你是一个任务规划专家。请将用户的任务分解为可执行的子步骤。"
        "返回严格的 JSON 格式：\n"
        '{"plan": [{"step": 1, "description": "...", "tool": "工具名", '
        '"depends_on": []}, ...]}\n'
        'depends_on 表示该步骤依赖哪些步骤（填写步骤号）。'
    )
    prompt = f"任务：{task}\n请制定执行计划。"
    return call_llm_with_json(prompt, system_msg)


"""
2.1.2 Reflexion 反思器实战
"""


def reflexion_review(original_answer: str, criteria: str) -> dict:
    """Reflexion 反思器：对已有的回答进行自我批评和改进。

    核心思想：
      Agent 完成一个动作后，让 LLM 自己审视结果：
      - 哪里做得好？
      - 哪里有不足？
      - 下次应该怎么改进？

    类比：像学生做了一道题后，自己用红笔批改并写反思。

    Args:
        original_answer: Agent 的原始输出。
        criteria: 评估标准。

    Returns:
        包含反思结果的字典。
    """
    system_msg = (
        "你是一个严格的质量审查员。请审查以下回答，"
        "指出问题和改进建议。返回严格的 JSON 格式：\n"
        '{"score": 1-10, "issues": ["问题1", "问题2"], '
        '"improved_answer": "改进后的回答"}'
    )
    prompt = (
        f"原始回答：\n{original_answer}\n\n"
        f"评估标准：{criteria}\n\n"
        f"请评估并改进。"
    )
    return call_llm_with_json(prompt, system_msg)


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.2 记忆系统 (Memory) —— Agent 的「大脑存储」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

记忆系统是 Agent 能力的核心差异点。

三类记忆（类比人类大脑）：

1. 短期记忆 (Short-term Memory)
   - 即当前对话的上下文（conversation history）
   - 实现：直接存在 messages 列表里
   - 局限：受 LLM 上下文窗口限制（如 128K tokens）

2. 长期记忆 (Long-term Memory)
   - 跨对话的持久化存储
   - 实现：向量数据库（ChromaDB / Pinecone / Milvus）
   - 用途：用户偏好、历史知识积累

3. 工作记忆 (Working Memory)
   - 当前任务的中间状态
   - 例：已完成的步骤、当前进度、收集到的中间数据
   - 实现：Python 变量、字典、外部缓存

面试常问：「如何解决 LLM 上下文窗口限制？」
答：
  1. 摘要压缩：用 LLM 将长对话总结为摘要
  2. 滑动窗口：只保留最近 N 轮对话
  3. 向量检索：将历史存入向量库，按相关性检索
  4. 混合策略：短期用滑动窗口，长期用向量检索 + 摘要
"""


class AgentMemory:
    """Agent 记忆系统 —— 模拟三种记忆类型。

    这是一个简化版实现，帮助理解每种记忆的本质。
    实际项目中用 LangChain 的 Memory 模块或向量数据库。
    """

    def __init__(self, max_short_term: int = 10):
        """
        Args:
            max_short_term: 短期记忆最多保留的对话轮数。
        """
        self.short_term = []          # 短期：最近对话 [(user, assistant), ...]
        self.working = {}             # 工作记忆：任务中间状态 {key: value}
        self.long_term = []           # 长期记忆（简化版）：重要信息列表
        self.max_short_term = max_short_term
        self.summary = ""             # 早期对话的摘要

    def add_conversation(self, user_msg: str, assistant_msg: str):
        """添加一轮对话到短期记忆。

        Args:
            user_msg: 用户消息。
            assistant_msg: Agent 回复。
        """
        self.short_term.append((user_msg, assistant_msg))

        # 超出限制时，压缩最早的对话为摘要
        if len(self.short_term) > self.max_short_term:
            self._compress()

    def _compress(self):
        """将最早的对话压缩为摘要，存入长期记忆。"""
        old_pair = self.short_term.pop(0)
        combined = f"[用户]: {old_pair[0]}\n[助手]: {old_pair[1]}"
        self.long_term.append(combined)
        if not self.summary:
            self.summary = f"早期对话摘要：\n{combined}"
        else:
            self.summary += f"\n...\n{combined}"

    def set_working(self, key: str, value: str):
        """写入工作记忆。

        Args:
            key: 键名。
            value: 值。
        """
        self.working[key] = value

    def get_working(self, key: str) -> str:
        """读取工作记忆。

        Args:
            key: 键名。

        Returns:
            对应的值，不存在则返回空字符串。
        """
        return self.working.get(key, "")

    def get_context_for_llm(self) -> str:
        """组装发给 LLM 的完整上下文。

        包含：长期记忆摘要 + 工作记忆 + 最近对话。

        Returns:
            格式化的上下文字符串。
        """
        parts = []

        if self.summary:
            parts.append(f"## 历史摘要\n{self.summary}\n")

        if self.working:
            wk_items = "\n".join(
                f"  {k}: {v}" for k, v in self.working.items()
            )
            parts.append(f"## 当前任务状态\n{wk_items}\n")

        if self.short_term:
            recent = "\n".join(
                f"用户: {u}\n助手: {a}"
                for u, a in self.short_term[-5:]
            )
            parts.append(f"## 最近对话\n{recent}")

        return "\n".join(parts)


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.3 工具设计 (Tool Design) —— Agent 的「武器库」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

工具是 Agent 与外部世界交互的唯一方式。
工具设计的好坏直接影响 Agent 的可用性。

工具设计的「4 条黄金法则」：

1. **描述清晰**：description 是 LLM 选择工具的唯一依据
   ✗ "查询天气"（太模糊）
   ✓ "查询指定城市当天的天气，返回温度、湿度、风向信息"

2. **职责单一**：一个工具只做一件事
   ✗ "search_and_summarize"（两个动作）
   ✓ 拆成 "search" 和 "summarize" 两个独立工具

3. **错误可读**：返回的错误信息要能帮助 LLM 修正调用
   ✗ return "Error: 404"
   ✓ return "未找到 '培京' 的天气数据。您是否指 '北京'？"

4. **参数限定**：用 enum 限制参数值，减少 LLM 幻觉
   parameters: {"city": {"enum": ["北京", "上海", "深圳"]}}

面试常问：「工具描述怎么写才有效？」
答：
  - 描述 LLM 什么时候该用这个工具（不是怎么实现）
  - 说明工具不擅长什么（边界条件）
  - 举例说明典型调用场景
"""


def demonstrate_tool_design():
    """演示高质量工具定义 vs 低质量工具定义。"""
    print("=" * 60)
    print("  工具设计对比：好 vs 坏")
    print("=" * 60)

    bad_tool = {
        "type": "function",
        "function": {
            "name": "do_stuff",
            "description": "处理数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string"}
                },
            },
        },
    }

    good_tool = {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "向指定邮箱发送邮件。适用于发送通知、报告、提醒等场景。"
                "不支持发送带附件的邮件。收件人地址必须包含 @。"
                "示例：send_email(to='user@example.com', "
                "subject='日报', body='今日数据...')"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "收件人邮箱地址，如 user@example.com",
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
    }

    print("\n❌ 低质量工具定义：")
    print(json.dumps(bad_tool, indent=2, ensure_ascii=False))

    print("\n✅ 高质量工具定义：")
    print(json.dumps(good_tool, indent=2, ensure_ascii=False))


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.4 综合演示：三大组件协同工作
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

下面演示一个完整流程：
  1. 规划器制定计划
  2. 记忆系统记录状态
  3. 执行器中调用工具
  4. 反思器自我审查
"""


def run_full_demo():
    """运行综合演示：组件协同工作。"""
    print("\n" + "█" * 60)
    print("█  综合演示：Agent 三大组件协同工作")
    print("█" * 60)

    # 初始化组件
    memory = AgentMemory(max_short_term=6)

    task = "调查北京和上海的天气，写一份简短对比报告"

    print(f"\n📋 任务: {task}")

    # 阶段 1：规划
    print("\n--- 阶段 1: 规划器制定计划 ---")
    plan = plan_execute(task)
    print(f"计划 {len(plan['plan'])} 个步骤：")
    for step in plan["plan"]:
        deps = f" (依赖步骤 {step['depends_on']})" if step.get("depends_on") else ""
        print(f"  步骤 {step['step']}: {step['description']}{deps}")
        memory.set_working(f"step_{step['step']}", step["description"])

    # 阶段 2：工具调用（模拟）
    print("\n--- 阶段 2: 工具调用（模拟）---")
    weather_data = {
        "北京": "晴天 25°C 湿度40%",
        "上海": "多云 28°C 湿度65%",
    }
    for city in ["北京", "上海"]:
        result = weather_data[city]
        memory.set_working(f"weather_{city}", result)
        print(f"  get_weather({city}) → {result}")

    # 阶段 3：生成回答
    print("\n--- 阶段 3: 生成对比报告（模拟）---")
    beijing = memory.get_working("weather_北京")
    shanghai = memory.get_working("weather_上海")
    answer = f"北京天气：{beijing}\n上海天气：{shanghai}\n建议：上海温度更高且湿度大，"
    answer += "北京更适合户外活动。"
    print(f"  初版回答:\n{answer}")

    # 阶段 4：反思
    print("\n--- 阶段 4: 反思器自我审查 ---")
    review = reflexion_review(
        answer,
        "应包含数值对比（温度差、湿度差）和明确的活动建议"
    )
    print(f"  评分: {review['score']}/10")
    print(f"  问题: {review['issues']}")
    print(f"\n  改进后回答:\n{review['improved_answer']}")

    memory.add_conversation(task, review['improved_answer'])

    # 展示记忆状态
    print("\n--- 记忆系统状态 ---")
    print(memory.get_context_for_llm())

    print("\n✅ 综合演示完成！")


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.5 本章总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心要点回顾：

1. 规划器 (Planner)
   - ReAct: 每步推理，灵活 (面试高频)
   - Plan-Execute: 先规划后执行，高效
   - Reflexion: 自我批评改进
   - 实际项目：组合使用

2. 记忆系统 (Memory)
   - 短期记忆: 对话上下文，受窗口限制
   - 长期记忆: 向量数据库持久化
   - 工作记忆: 当前任务的中间状态
   - 面试重点：上下文窗口管理策略

3. 工具设计 (Tool Design)
   - 描述要详细、准确
   - 职责要单一
   - 错误信息要可读
   - 这是区分初级/高级 Agent 工程师的关键技能

面试速记：
  "请描述 Agent 的核心架构"
  → Agent = LLM + Planner + Memory + Tools
  → 通过 ReAct 循环协调各组件工作
  → 记忆系统解决上下文限制，工具系统扩展能力边界
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║       第2章：Agent 核心组件深度解析                    ║")
    print("║       规划器 · 记忆系统 · 工具设计                    ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n▶ 2.1-2.2: 规划器示例（Plan-Execute）")
    plan = plan_execute("分析特斯拉股票是否值得投资")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    print("\n▶ 2.3: 工具设计对比演示")
    demonstrate_tool_design()

    print("\n▶ 2.4: 综合演示（需要 API 调用）")
    try:
        run_full_demo()
    except Exception as e:
        print(f"\n⚠️ 综合演示需要 API Key，错误信息: {e}")
        print("请确保 .env 文件配置正确后重试。")
