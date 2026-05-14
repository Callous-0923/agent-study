"""
第19章：Agentic Workflow 设计模式 —— 系统设计的万能框架
========================================================

📌 本章目标：
  1. 掌握 7 种核心 Agentic 设计模式及其适用场景
  2. 理解每种模式的执行流程、代价和权衡
  3. 能用 LangGraph 实现关键模式
  4. 建立系统设计题的模式选择框架
  5. 学会避免 Top 3 常见错误

📌 面试高频点：
  - 「你用过哪些 Agent 设计模式？」—— 必须能说出 4+ 种并对比
  - 「Reflection 模式怎么实现？和纯 LLM 有什么区别？」
  - 「Routing 和 Orchestrator-Worker 的区别？」
  - 「什么时候不该用 Agent？」—— Anthropic 核心理念

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
综合 Andrew Ng (4模式) + Anthropic (5模式) + 业界 (7模式) 体系
参考: https://www.anthropic.com/engineering/building-effective-agents
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


19.1 Workflows vs Agents —— 先搞清楚概念
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Anthropic 的核心区分（面试必考！）：

  Workflows（工作流）：
    LLM 按预定义路径执行，由代码控制流程。
    → 步骤可预测时使用

  Agents（智能体）：
    LLM 动态决定自己的流程和工具使用。
    → 步骤不可预测时使用

关键原则（面试金句）：
  "When building applications with LLMs, we recommend finding the
   simplest solution possible, and only increasing complexity when
   needed. This might mean not building agentic systems at all."

19.2 设计模式全景图
━━━━━━━━━━━━━━━━━━

┌──────────────────────────────────────────────────────────────┐
│                  Agentic Design Patterns                       │
├────────────┬─────────────────────────────────────────────────┤
│ 1. ReAct   │ 推理-行动循环，最通用（Ch3/Ch8 已深入讲过）       │
│ 2. Reflection│ 自我批评 → 迭代改进（本章19.3）                │
│ 3. Routing  │ 按任务特征路由到不同处理器（本章19.4）           │
│ 4. Planning │ 先制定计划再执行（本章19.5）                    │
│ 5. Orchestrator│ 协调者分发任务给专业 Worker（本章19.6）       │
│ 6. Evaluator │ 评估-优化双循环（本章19.7）                   │
│ 7. Human-in- │ 关键节点人工确认（本章19.8）                  │
│    the-Loop │                                                 │
└────────────┴─────────────────────────────────────────────────┘

快速选择矩阵（面试时脱口而出！）：
┌────────────────────────────┬──────────────────────────────┐
│         任务特征             │         推荐模式              │
├────────────────────────────┼──────────────────────────────┤
│ 输出质量 > 速度             │ Reflection                   │
│ 多种类型但可分类的任务       │ Routing                      │
│ 复杂的多步骤任务            │ Planning + ReAct             │
│ 可分解为独立子任务          │ Orchestrator-Worker          │
│ 需要通过评测指标迭代优化    │ Evaluator-Optimizer          │
│ 涉及资金/数据安全的操作     │ Human-in-the-Loop            │
│ 路径不可预测的复杂推理      │ 纯 ReAct Agent               │
│ 简单、步骤固定的任务        │ 不需要 Agent！直接用 LLM 调用  │
└────────────────────────────┴──────────────────────────────┘
"""

import json
import time
import re
from typing import Optional
from dataclasses import dataclass, field


"""
19.3 Reflection 模式 —— 自我批评，迭代改进
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心理念：
  不满足于 LLM 的第一次输出，让 LLM 自己批评自己，然后改进。

流程：
  初版生成 → 自我审查 → 发现不足 → 修订 → 再审查 → ... → 达标输出

为什么「双 Agent」比「单 Agent 自省」更好？
  - LLM 对自己的输出有「确认偏误」
  - 换成另一个 Agent（不同的 system prompt）来审查更客观
  - Generator + Critic 两个角色分离

类比：
  就像作家写完初稿后，自己很难发现错字
  需要一个编辑来审阅
"""


class ReflectionAgent:
    """Reflection 模式实现 —— Generator + Critic 双角色。

    流程：
      Generator 生成 → Critic 审查 → 如果不够好 → Generator 修订 → ...
    """

    def __init__(self, quality_threshold: int = 8, max_rounds: int = 3):
        """
        Args:
            quality_threshold: 质量阈值（1-10），达到即停止。
            max_rounds: 最大迭代轮数。
        """
        self.threshold = quality_threshold
        self.max_rounds = max_rounds

    def generate(self, task: str, previous: str = "",
                 feedback: str = "") -> str:
        """Generator：生成/修订输出（模拟）。"""
        if not previous:
            return self._generate_initial(task)
        return self._revise(task, previous, feedback)

    def critique(self, output: str, task: str) -> dict:
        """Critic：以角色分离的方式审查输出。

        真实实现中用不同的 system prompt 或不同的模型。
        """
        issues = []
        score = 10

        # 规则检查（模拟 LLM-as-Critic）
        if len(output) < 50:
            issues.append("输出太短，缺乏细节")
            score -= 3

        if task.lower() in output.lower():
            pass  # 回应了任务
        else:
            issues.append("没有直接回应任务要求")
            score -= 2

        # 检查是否包含推测性语言
        uncertain_patterns = ["可能", "也许", "大概", "似乎"]
        uncertain_count = sum(
            1 for p in uncertain_patterns if p in output
        )
        if uncertain_count >= 3:
            issues.append(f"包含 {uncertain_count} 处不确定表述")
            score -= 1

        # 检查结构
        if "\n" not in output or len(output.split("\n")) < 3:
            issues.append("缺少段落结构，可读性差")
            score -= 1

        return {
            "score": max(1, score),
            "issues": issues,
            "passed": score >= self.threshold,
        }

    def run(self, task: str) -> dict:
        """执行完整的 Reflection 循环。

        Returns:
            包含最终输出和迭代记录的字典。
        """
        history = []
        output = self.generate(task)

        for round_num in range(1, self.max_rounds + 1):
            review = self.critique(output, task)
            history.append({
                "round": round_num,
                "score": review["score"],
                "issues": review["issues"],
            })

            if review["passed"]:
                break

            if round_num < self.max_rounds:
                feedback = "；".join(review["issues"])
                output = self.generate(
                    task, previous=output, feedback=feedback
                )

        return {
            "task": task,
            "final_output": output,
            "history": history,
            "rounds": len(history),
            "passed": history[-1]["score"] >= self.threshold,
        }

    def _generate_initial(self, task: str) -> str:
        return f"关于「{task}」，AI Agent 是一种能够自主感知环境、" \
               f"做出决策并执行行动的智能系统。"

    def _revise(self, task: str, previous: str, feedback: str) -> str:
        return f"修订版（收到反馈: {feedback}）：\n{previous}\n" \
               f"补充说明：AI Agent 主要由 LLM、规划器、记忆系统和" \
               f"工具调用四部分组成，通过 ReAct 循环协调工作。"


"""
19.4 Routing 模式 —— 分类 → 分发
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心理念：
  不同的任务需要不同的处理方式，先用 LLM 分类再路由。

流程：
  用户输入 → 分类 Agent → 路由到专门的 Handler → 返回结果

适用场景：
  - 客服系统（投诉/咨询/售后 分到不同的 Agent）
  - 代码工具（需求/调试/代码审查 分到不同的流程）
  - 智能助手（简单问答/复杂推理/工具调用 分到不同的模式）

关键决策：
  - 路由的依据是什么？（关键词/语义/复杂度评分？）
  - 每个路由目标用不同模型、不同 prompt、不同工具集
  - 可以实现成本优化（简单问题用小模型，复杂问题用大模型）
"""


class RoutingAgent:
    """Routing 模式 —— 根据任务特征选择最合适的处理器。"""

    ROUTE_TABLE = {
        "简单问答": {"model": "小模型", "max_tokens": 256},
        "代码相关": {"model": "大模型", "max_tokens": 2048, "tools": ["code_exec"]},
        "数据分析": {"model": "大模型", "max_tokens": 4096, "tools": ["sql", "chart"]},
        "复杂推理": {"model": "大模型", "max_tokens": 8192, "tools": ["search", "calculator"]},
        "安全敏感": {"model": "审核模型", "human_review": True},
    }

    @staticmethod
    def classify(task: str) -> str:
        """分类器：分析任务特征，决定路由目标。

        真实实现中用 LLM 做分类：
          prompt = f"将以下任务分类为 [简单问答/代码相关/数据分析/复杂推理/安全敏感]: {task}"
        """
        task_lower = task.lower()

        if any(w in task_lower for w in ["代码", "debug", "写一个", "实现"]):
            return "代码相关"
        if any(w in task_lower for w in ["分析", "统计", "趋势", "报表", "数据"]):
            return "数据分析"
        if any(w in task_lower for w in ["删除", "密码", "转账", "admin"]):
            return "安全敏感"
        if len(task) > 100 or any(
            w in task_lower for w in ["为什么", "如何", "解释", "原因"]
        ):
            return "复杂推理"
        return "简单问答"

    @staticmethod
    def route(task: str) -> dict:
        """路由：分类 → 选择配置 → 返回处理方案。

        Returns:
            包含路由决策的字典。
        """
        category = RoutingAgent.classify(task)
        config = RoutingAgent.ROUTE_TABLE[category].copy()
        config["category"] = category
        config["task"] = task
        config["estimated_cost"] = _estimate_cost(config)
        return config


def _estimate_cost(config: dict) -> float:
    """估算处理成本（模拟）。"""
    base = {"简单问答": 0.001, "代码相关": 0.01,
            "数据分析": 0.05, "复杂推理": 0.10, "安全敏感": 0.02}
    return base.get(config["category"], 0.01)


"""
19.5 Planning 模式 —— 先规划，再执行
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

和 Ch3 中 Plan-Execute 的关系：
  这是更高层的抽象，规划出来的每个子步骤可以由 ReAct Agent 执行。

核心理念：
  复杂任务 → 分解为子步骤 → 每个子步骤独立执行 → 汇总结果

关键洞察：
  1. 规划用 LLM（灵活），执行按计划（可靠）
  2. 规划不是「一次性」的 —— 执行中可以根据反馈调整计划
  3. 规划质量取决于分解粒度（太粗执行不了，太细浪费 token）
"""


class PlanningAgent:
    """Planning 模式 —— 任务分解 + 分步执行。"""

    @staticmethod
    def decompose(task: str) -> list[dict]:
        """任务分解：将复杂任务拆成子步骤。

        真实实现用 LLM：
          f"将以下任务分解为 3-6 个可独立执行的子步骤: {task}"
        """
        # 模拟分解逻辑
        if "分析" in task and "写" in task:
            return [
                {"id": 1, "action": "research", "desc": "搜索相关资料"},
                {"id": 2, "action": "analyze", "desc": "分析关键信息",
                 "depends_on": [1]},
                {"id": 3, "action": "outline", "desc": "制定文章大纲",
                 "depends_on": [2]},
                {"id": 4, "action": "write", "desc": "撰写初稿",
                 "depends_on": [3]},
                {"id": 5, "action": "review", "desc": "自我审查并修订",
                 "depends_on": [4]},
            ]
        if "搭建" in task or "构建" in task:
            return [
                {"id": 1, "action": "analyze", "desc": "分析需求和约束"},
                {"id": 2, "action": "design", "desc": "设计架构",
                 "depends_on": [1]},
                {"id": 3, "action": "implement", "desc": "编码实现",
                 "depends_on": [2]},
                {"id": 4, "action": "test", "desc": "编写测试",
                 "depends_on": [3]},
                {"id": 5, "action": "document", "desc": "编写文档",
                 "depends_on": [4]},
            ]
        return [
            {"id": 1, "action": "execute", "desc": task},
        ]

    @staticmethod
    def execute_plan(plan: list[dict]) -> list[dict]:
        """按计划执行（模拟）。

        真实实现中每个步骤调用 ReAct Agent 或直接调 LLM。
        """
        results = []
        for step in plan:
            deps = step.get("depends_on", [])
            # 检查依赖是否完成
            dep_results = [r for r in results if r["id"] in deps]
            if deps and len(dep_results) != len(deps):
                results.append({
                    **step, "status": "blocked",
                    "reason": f"依赖步骤 {deps} 未完成",
                })
                continue

            # 模拟执行
            results.append({
                **step,
                "status": "completed",
                "output": f"已完成: {step['desc']}",
            })
        return results


"""
19.6 Orchestrator-Worker 模式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心理念：
  一个聪明的 Orchestrator（有状态）管理多个无状态 Worker。
  Worker 之间不直接通信，通过 Orchestrator 协调。

这是 Claude Code 1.0.60+ SubAgent 系统的本质（参见 Ch8）。

对比 Planning 模式：
  Planning：预先分解，固定执行
  Orchestrator-Worker：动态分配，Worker 独立运行

关键特性：
  1. Orchestrator 有状态（记住任务上下文）
  2. Worker 无状态（每次只处理一个子任务）
  3. Worker 之间隔离（可以并行，可以不同权限）
  4. 易于水平扩展（加 Worker 即可）
"""


class OrchestratorWorker:
    """Orchestrator-Worker 模式实现。"""

    def __init__(self):
        self.workers = {}  # {name: {permissions, handler}}
        self.task_state = {}  # 全局任务状态

    def register_worker(self, name: str, permissions: list[str]):
        """注册一个 Worker。

        Args:
            name: Worker 名称。
            permissions: 拥有的权限列表。
        """
        self.workers[name] = {
            "permissions": permissions,
            "status": "idle",
            "completed_tasks": 0,
        }

    def orchestrate(self, task: str) -> dict:
        """Orchestrator：分析任务 → 分解 → 分配给 Worker。

        Returns:
            完整的执行记录。
        """
        # 第1步：Orchestrator 分析任务
        subtasks = self._decompose(task)

        # 第2步：分配给 Worker
        results = []
        for sub in subtasks:
            worker = self._select_worker(sub)
            if worker is None:
                results.append({**sub, "status": "unassigned"})
                continue

            # Worker 执行（模拟）
            self.workers[worker]["status"] = "working"
            result = self._execute(sub, worker)
            self.workers[worker]["status"] = "idle"
            self.workers[worker]["completed_tasks"] += 1
            results.append(result)

        return {
            "task": task,
            "subtasks": len(subtasks),
            "results": results,
            "workers_used": len(set(r.get("worker") for r in results)),
        }

    def _decompose(self, task: str) -> list[dict]:
        """Orchestrator 分解任务。"""
        if "代码" in task and "文档" in task:
            return [
                {"id": 1, "type": "code", "desc": "编写代码"},
                {"id": 2, "type": "docs", "desc": "编写文档"},
                {"id": 3, "type": "test", "desc": "运行测试"},
            ]
        if "全栈" in task:
            return [
                {"id": 1, "type": "frontend", "desc": "前端开发"},
                {"id": 2, "type": "backend", "desc": "后端开发"},
                {"id": 3, "type": "database", "desc": "数据库设计"},
            ]
        return [{"id": 1, "type": "general", "desc": task}]

    def _select_worker(self, subtask: dict) -> Optional[str]:
        """根据子任务类型选择合适的 Worker。"""
        type_map = {
            "code": "coder", "docs": "writer", "test": "tester",
            "frontend": "frontend_dev", "backend": "backend_dev",
            "database": "db_admin", "general": "general_assistant",
        }
        target = type_map.get(subtask["type"], "general_assistant")
        return target if target in self.workers else None

    def _execute(self, subtask: dict, worker: str) -> dict:
        return {
            **subtask,
            "worker": worker,
            "permissions": self.workers[worker]["permissions"],
            "status": "completed",
            "output": f"[{worker}] 已完成: {subtask['desc']}",
        }


"""
19.7 Evaluator-Optimizer 模式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心理念：
  和 Reflection 类似但不同：
  - Reflection: 同一 LLM 生成 + 批评
  - Evaluator-Optimizer: 独立评测 + 独立优化，评测标准更客观

适用场景：
  - 翻译质量优化（BLEU 评分驱动）
  - 代码性能优化（Benchmark 评分驱动）
  - Prompt 自动优化（准确率评分驱动）

这个模式是 DSPy（Ch22 待讲）的核心思想。
"""


"""
19.8 Human-in-the-Loop 模式
━━━━━━━━━━━━━━━━━━━━━━━━━━

核心理念：
  不是所有操作都应该自动执行。在关键节点引入人工确认。

触发时机：
  ✗ 资金操作（转账、下单）
  ✗ 内容发布（发送邮件、发布文章）
  ✗ 数据删除
  ✗ 系统配置变更
  ✓ 查询类操作 → 自动执行

这是 Ch18 Agent 安全的分级确认机制的扩展。
"""


"""
19.9 Top 3 常见错误（Tina Huang + Anthropic 总结）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

错误 1: 过度工程化 (Over-Engineering)
  症状：用 Multi-Agent 做单步就能完成的事
  解法：先问「这真的需要 Agent 吗？」

错误 2: 糟糕的 Prompt (Bad Prompting)
  症状：复杂架构但 prompt 写得很随意
  解法：先投入时间优化 prompt，再考虑架构

错误 3: 糟糕的系统设计 (Poor System Design)
  症状：没有架构图、数据流不清晰、难以调试
  解法：选好架构模式 → 画架构图 → 定义数据流 → 设置错误处理


19.10 系统设计题的答题框架
━━━━━━━━━━━━━━━━━━━━━━━━━━

面试官：「设计一个智能客服系统」

标准回答结构：
  1. 场景分析（任务特征分类）
  2. 模式选择（主模式 + 辅助模式）
  3. 架构设计（画图！）
  4. 关键决策说明（为什么选这个模式）
  5. 风险与权衡

示例回答：
  「智能客服有3类请求：FAQ（70%）、工单操作（20%）、投诉（10%）。
   用 Routing 模式分流：FAQ 走 RAG + 小模型；工单走 Tool Use；
   投诉需要升级（Human-in-the-Loop）。
   在工单处理中用 Reflection 确保回复质量。
   敏感操作（退款）需要双重确认。」
"""


def demo_all_patterns():
    """演示所有设计模式。"""
    print("=" * 60)
    print("  Agentic Workflow 设计模式综合演示")
    print("=" * 60)

    # Reflection
    print("\n  ── Reflection 模式 ──")
    agent = ReflectionAgent(quality_threshold=8, max_rounds=3)
    result = agent.run("解释 AI Agent 的核心组成")
    print(f"  任务: {result['task']}")
    print(f"  轮数: {result['rounds']}")
    for h in result["history"]:
        icon = "✅" if h["score"] >= 8 else "🔄"
        print(f"    {icon} 第{h['round']}轮: {h['score']}/10 | {h['issues']}")

    # Routing
    print("\n  ── Routing 模式 ──")
    test_tasks = [
        "帮我写一个 Python 快速排序函数",
        "分析最近一个月的销售数据趋势",
        "什么是机器学习？",
        "删除用户 admin 的所有数据",
    ]
    for task in test_tasks:
        decision = RoutingAgent.route(task)
        print(f"  「{task[:30]}...」→ {decision['category']} "
              f"| 模型: {decision.get('model','')} "
              f"| 成本: ${decision.get('estimated_cost', 0):.3f}")

    # Planning
    print("\n  ── Planning 模式 ──")
    plan = PlanningAgent.decompose("分析AI Agent市场并写一份报告")
    print(f"  分解为 {len(plan)} 个步骤:")
    for step in plan:
        deps = f" (依赖步骤 {step['depends_on']})" if step.get("depends_on") else ""
        print(f"    Step {step['id']}: {step['desc']}{deps}")

    # Orchestrator-Worker
    print("\n  ── Orchestrator-Worker 模式 ──")
    ow = OrchestratorWorker()
    ow.register_worker("coder", ["read", "write", "execute"])
    ow.register_worker("writer", ["read", "write"])
    ow.register_worker("tester", ["execute"])
    result = ow.orchestrate("开发一个功能并写文档，确保代码和文档都完整")
    print(f"  子任务: {result['subtasks']} | 使用 Worker: {result['workers_used']}")
    for r in result["results"]:
        print(f"    [{r.get('worker', '?')}] {r['desc']} → {r['status']}")


"""
19.11 本章总结
━━━━━━━━━━━━━━

核心要点回顾：

1. Workflow ≠ Agent
   - Workflow: 代码控制流程
   - Agent: LLM 控制流程
   - 能不用 Agent 就不用（Anthropic 核心原则）

2. 7 种核心模式
   - ReAct: 最通用，默认选择
   - Reflection: 质量优先场景
   - Routing: 多种任务类型
   - Planning: 复杂多步骤
   - Orchestrator-Worker: 可并行子任务
   - Evaluator-Optimizer: 量化优化
   - Human-in-the-Loop: 安全关键操作

3. 系统设计答题框架
   场景分析 → 模式选择 → 架构设计 → 关键决策 → 风险权衡

面试速记：
  "你用过哪些 Agent 设计模式？"
  → 说出 4+ 种 + 每种的使用场景 + 对比
  → 重点说 Reflection（双角色分离）和 Routing（成本优化）
  → 结尾：Anthropic 原则「简单优先，够用就好」
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第19章：Agentic Workflow 设计模式                     ║")
    print("║  Reflection · Routing · Planning · Orchestrator      ║")
    print("╚══════════════════════════════════════════════════════╝")

    demo_all_patterns()

    print("\n▶ 模式选择速查表")
    print("-" * 50)
    choices = [
        ("输出质量 > 速度", "Reflection"),
        ("多种可分类任务", "Routing"),
        ("复杂多步骤任务", "Planning + ReAct"),
        ("可并行子任务", "Orchestrator-Worker"),
        ("量化优化", "Evaluator-Optimizer"),
        ("安全关键操作", "Human-in-the-Loop"),
        ("简单固定步骤", "直接 LLM（不用 Agent）"),
    ]
    for scene, pattern in choices:
        print(f"  {scene:20s} → {pattern}")

    print("\n✅ 第19章完成！")
