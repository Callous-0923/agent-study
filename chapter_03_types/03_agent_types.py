"""
第3章：Agent 类型分类与设计模式
=================================

📌 本章目标：
  1. 掌握 5 种主流 Agent 类型的设计模式与适用场景
  2. 理解每种模式的执行流程和优劣对比
  3. 能根据任务特征选择合适的 Agent 类型
  4. 实现一个可切换模式的 Agent 实验框架

📌 面试高频点：
  - ReAct 的执行流程（面试必问！）
  - Plan-Execute vs ReAct 的取舍
  - Reflexion 的反思机制
  - 什么场景用哪种 Agent 类型


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent 类型全景图
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  1. ReAct Agent        → 每步推理，最通用          │
  │  2. Plan-Execute Agent → 先规划再执行，结构化任务   │
  │  3. Reflexion Agent    → 执行+反思，自我改进       │
  │  4. Tool-calling Agent → 纯工具调用，不需要推理     │
  │  5. Multi-Agent        → 多角色协作（第5章详讲）    │
  │                                                    │
  └────────────────────────────────────────────────────┘

各类型关键区别：
┌──────────────┬──────────┬──────────┬──────────┬──────────┐
│     特性      │  ReAct   │Plan-Exec │Reflexion │Tool-Call │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 规划方式      │ 动态推理  │ 前置规划  │ 动态+反思 │ 不规划   │
│ 复杂任务      │ 中等     │ 强        │ 强        │ 弱       │
│ 执行效率      │ 中等     │ 高        │ 低(多次)  │ 高       │
│ 错误恢复      │ 自动     │ 需重新规划 │ 强(反思)  │ 弱       │
│ Token 消耗    │ 中等     │ 低        │ 高        │ 低       │
│ 典型场景      │ 通用     │ 数据分析  │ 代码审查  │ API 网关 │
└──────────────┴──────────┴──────────┴──────────┴──────────┘

面试速记：
  "请对比 ReAct 和 Plan-Execute"
  → ReAct 灵活但费 token，Plan-Execute 高效但不灵活
  → 大型任务用 Plan-Execute 做顶层规划，ReAct 做每步执行
  → Google 的 ReAct 论文是 Agent 领域最重要论文之一
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# 共享工具集
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "在互联网上搜索信息。适用于获取实时信息、事实查询等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算。支持加减乘除、乘方等运算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
]

TOOL_MAP = {
    "search": lambda query: f"搜索结果(模拟): 关于'{query}'的信息...",
    "calculator": lambda expression: f"计算结果(模拟): {eval(expression) if all(c in '0123456789+-*/().% ' for c in expression) else '表达式错误'}",
}


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3.1 ReAct Agent（面试必问！）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ReAct = Reasoning + Acting（推理 + 行动）

执行流程：
  Thought → Action → Observation → Thought → Action → ... → Final Answer

每次循环：
  1. Thought: "我需要知道 X" → 推理
  2. Action: 调用 search("X") → 行动
  3. Observation: 收到搜索结果 → 观察
  → 基于观察继续思考，直到能给出最终答案

类比：
  就像你在解决一道复杂数学题——
  你不可能一次算出结果，而是：
  "先化简...然后代入...发现不对...换个思路..."
  每一次"想 → 做 → 看结果"就是一个 ReAct 循环。

论文来源：
  Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models"
  (ICLR 2023) — Agent 领域的奠基性论文
"""


class ReActAgent:
    """ReAct Agent 实现 —— 最经典的 Agent 模式。

    核心算法：
      while not done:
          response = llm(messages + tools)
          if tool_calls:
              execute_tools(response.tool_calls)
              append results to messages
          else:
              return response.content
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """初始化 ReAct Agent。

        Args:
            system_prompt: 自定义系统提示词。
        """
        self.system_prompt = system_prompt or (
            "你是一个有用的助手。面对需要实时信息或计算的问题，"
            "请使用提供的工具。推理过程请在思考后行动。"
        )

    def run(self, user_message: str, max_steps: int = 5) -> str:
        """执行 ReAct 循环。

        Args:
            user_message: 用户消息。
            max_steps: 最大步数限制。

        Returns:
            Agent 的最终回答。
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for step in range(max_steps):
            print(f"\n  [ReAct 第 {step+1} 步]")

            response = client.chat.completions.create(
                model=MODEL, messages=messages,
                tools=TOOLS, tool_choice="auto",
            )
            msg = response.choices[0].message

            if msg.tool_calls is None:
                print(f"  → 最终回答")
                return msg.content

            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"  → 调用: {name}({args})")
                result = TOOL_MAP[name](**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "达到最大步数限制，无法完成任务。"


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3.2 Plan-Execute Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

「先规划，再执行」—— 像项目经理一样工作。

执行流程：
  1. 分析任务 → 制定详细执行计划
  2. 按计划逐步执行每个子任务
  3. 汇总各步骤结果 → 生成最终答案

适用场景：
  - 任务步骤明确、有依赖关系
  - 数据分析报表生成
  - 代码审查流水线
  - 工作流自动化
"""


class PlanExecuteAgent:
    """Plan-Execute Agent —— 先规划再逐步执行。"""

    def __init__(self):
        self.plan = []
        self.results = {}

    def _make_plan(self, task: str) -> list:
        """调用 LLM 制定执行计划。

        Args:
            task: 用户任务描述。

        Returns:
            计划步骤列表。
        """
        prompt = (
            f"请将以下任务分解为 2-4 个可执行的子步骤。"
            f"返回 JSON: {{\"steps\": [{{\"id\": 1, \"desc\": \"...\", "
            f"\"tool\": \"search|calculator|none\", "
            f"\"tool_input\": \"...\"}}]}}\n\n任务: {task}"
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("steps", [])

    def run(self, task: str) -> str:
        """执行 Plan-Execute 流程。

        Args:
            task: 用户任务描述。

        Returns:
            执行结果汇总。
        """
        print("\n  [Plan-Execute] 阶段 1: 制定计划")
        self.plan = self._make_plan(task)

        for step in self.plan:
            print(f"    步骤 {step['id']}: {step['desc']}")
            print(f"      工具: {step['tool']}")

        print("\n  [Plan-Execute] 阶段 2: 执行计划")
        for step in self.plan:
            tool = step.get("tool", "none")
            tool_input = step.get("tool_input", "")
            if tool == "none":
                result = f"已完成: {step['desc']}"
            elif tool in TOOL_MAP:
                result = TOOL_MAP[tool](tool_input)
            else:
                result = f"暂不支持的工具: {tool}"
            self.results[f"step_{step['id']}"] = result
            print(f"    步骤 {step['id']} 结果: {result}")

        print("\n  [Plan-Execute] 阶段 3: 生成汇总")
        summary_prompt = (
            f"任务: {task}\n"
            f"执行结果:\n{json.dumps(self.results, ensure_ascii=False)}\n"
            f"请基于以上结果给出最终回答。"
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        return response.choices[0].message.content


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3.3 Reflexion Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

「做一遍 → 反思 → 改进 → 再做一遍」—— 自我迭代优化。

执行流程：
  1. 初次执行任务
  2. 自我评估：打分 + 找问题
  3. 基于反馈重新执行
  4. 重复直到满意

核心洞察：
  Reflexion 不改变 Agent 的基础能力，
  而是通过「反馈循环」让 Agent 在实践中变得更好。
  就像学生学习：做题 → 对答案 → 发现错在哪里 → 下次改进。

论文来源：
  Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning"
  (NeurIPS 2023)
"""


class ReflexionAgent:
    """Reflexion Agent —— 执行 + 自我反思 + 迭代改进。"""

    def __init__(self, max_reflections: int = 3):
        """
        Args:
            max_reflections: 最大反思迭代次数。
        """
        self.max_reflections = max_reflections

    def _evaluate(self, answer: str, task: str) -> dict:
        """自我评估回答质量。

        Args:
            answer: Agent 的回答。
            task: 原始任务。

        Returns:
            包含评分和问题的字典。
        """
        prompt = (
            f"任务: {task}\n"
            f"回答: {answer}\n\n"
            f"请从准确性、完整性、清晰度三方面评分(1-10)，"
            f"并指出具体问题。返回 JSON:\n"
            f'{{"accuracy": 0, "completeness": 0, "clarity": 0, '
            f'"issues": ["问题1"]}}'
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def run(self, task: str) -> str:
        """执行 Reflexion 流程。

        Args:
            task: 用户任务描述。

        Returns:
            经过反思迭代后的最佳回答。
        """
        print(f"\n  [Reflexion] 任务: {task}")
        feedback_history = []

        for iteration in range(self.max_reflections):
            print(f"\n  [Reflexion] 第 {iteration+1} 轮")

            # 构建带反馈的提示词
            if feedback_history:
                feedback_text = "\n".join(
                    f"上一轮问题 {i+1}: {fb}"
                    for i, fb in enumerate(feedback_history[-1])
                )
                prompt = (
                    f"任务: {task}\n"
                    f"请改进之前的回答。之前的问题:\n{feedback_text}"
                )
            else:
                prompt = task

            # 执行（不调用工具，纯推理）
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content
            print(f"    回答: {answer[:100]}...")

            # 自我评估
            evaluation = self._evaluate(answer, task)
            avg_score = (
                evaluation.get("accuracy", 0)
                + evaluation.get("completeness", 0)
                + evaluation.get("clarity", 0)
            ) / 3
            issues = evaluation.get("issues", [])
            feedback_history.append(issues)
            print(f"    评分: {avg_score:.1f}/10 | 问题数: {len(issues)}")

            # 质量达标则停止
            if avg_score >= 8.0:
                print(f"    质量达标，停止迭代！")
                return answer

        return answer


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3.4 Agent 模式选择器
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

面试常见问题：「如何为任务选择合适的 Agent 类型？」

快速判断矩阵：
┌──────────────────────────────────┬──────────────────────┐
│           任务特征               │     推荐 Agent 类型    │
├──────────────────────────────────┼──────────────────────┤
│ 需要实时搜索/计算                │ Tool-Calling          │
│ 开放式推理 + 工具调用            │ ReAct                 │
│ 步骤明确、可预先规划             │ Plan-Execute          │
│ 需要高质量输出、可迭代           │ Reflexion             │
│ 多角色协作                       │ Multi-Agent           │
│ 简单问答、不需要工具             │ 直接 LLM 调用即可     │
└──────────────────────────────────┴──────────────────────┘
"""

AGENT_REGISTRY = {
    "react": ReActAgent,
    "plan_execute": PlanExecuteAgent,
    "reflexion": ReflexionAgent,
}


def run_agent_by_type(agent_type: str, task: str):
    """统一的 Agent 运行入口。

    Args:
        agent_type: Agent 类型（react / plan_execute / reflexion）。
        task: 用户任务。

    Returns:
        Agent 的执行结果。
    """
    agent_class = AGENT_REGISTRY.get(agent_type)
    if agent_class is None:
        return f"不支持的 Agent 类型: {agent_type}。支持: {list(AGENT_REGISTRY.keys())}"

    print(f"\n{'='*60}")
    print(f"  Agent 类型: {agent_type.upper()}")
    print(f"  任务: {task}")
    print(f"{'='*60}")

    agent = agent_class()
    result = agent.run(task)
    print(f"\n  ✅ 最终答案:\n{result}")
    return result


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3.5 本章总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心要点回顾：

1. ReAct (面试必问！)
   - 推理 + 行动循环
   - 最通用、最灵活
   - 缺点：每步都调用 LLM，token 消耗大

2. Plan-Execute
   - 先规划后执行
   - 适合结构化任务
   - 缺点：计划不合理时需要重来

3. Reflexion
   - 自我反思迭代
   - 适合质量要求高的任务
   - 缺点：多次 LLC 调用，成本高

4. 实际项目中往往是组合使用
   - 高层 Plan-Execute 做任务分解
   - 每层 ReAct 做具体执行
   - 关键步骤加 Reflexion 做质量把关

面试常问：
  "你在项目中用的是什么 Agent 架构？为什么选择它？"
  → 回答框架：场景特征 → 选型理由 → 实际效果
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║       第3章：Agent 类型分类与设计模式                  ║")
    print("║       ReAct · Plan-Execute · Reflexion               ║")
    print("╚══════════════════════════════════════════════════════╝")

    test_task = "分析 Python 和 JavaScript 各自的优势，给出学习建议。"

    print("\n▶ 3.1 测试 ReAct Agent")
    try:
        result = run_agent_by_type("react", test_task)
    except Exception as e:
        print(f"  ⚠️ ReAct 测试需要 API Key: {e}")

    print("\n▶ 3.2 测试 Plan-Execute Agent")
    try:
        result = run_agent_by_type("plan_execute", test_task)
    except Exception as e:
        print(f"  ⚠️ Plan-Execute 测试需要 API Key: {e}")

    print("\n▶ 3.3 测试 Reflexion Agent")
    try:
        result = run_agent_by_type("reflexion", test_task)
    except Exception as e:
        print(f"  ⚠️ Reflexion 测试需要 API Key: {e}")

    print("\n▶ 3.4 Agent 类型快速选择指南")
    print("┌─────────────┬──────────────────────────────────────┐")
    print("│   任务特征    │              推荐类型                 │")
    print("├─────────────┼──────────────────────────────────────┤")
    print("│ 搜索+计算    │  Tool-Calling / ReAct                │")
    print("│ 开放推理     │  ReAct                                │")
    print("│ 结构化任务   │  Plan-Execute                         │")
    print("│ 高质量要求   │  Reflexion                            │")
    print("│ 多角色协作   │  Multi-Agent (第5章)                  │")
    print("│ 简单问答     │  直接 LLM 调用                        │")
    print("└─────────────┴──────────────────────────────────────┘")

    print("\n✅ 第3章完成！接下来进入第4章：主流框架实战。")
