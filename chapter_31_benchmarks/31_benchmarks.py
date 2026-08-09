"""
第31章：Agent 评测体系深度 —— 不只是「看起来不错」
=====================================================

内容核对：2026-08-01
说明：标注为模拟的实现与数值用于讲解概念，不代表真实 SDK、协议或基准结果。

📌 本章目标：
  1. 掌握 GAIA / AgentBench / WebArena / tau-bench / SWE-bench 五大评测
  2. 理解每个评测的设计意图和适用场景
  3. 学会自建 Agent 评测集的方法
  4. 建立生产环境的多维度评测体系

📌 面试高频点：
  - 「你们怎么评测 Agent 的质量？」
  - 「GAIA 和 SWE-bench 有什么区别？」
  - 「怎么自建评测集？」

公开 benchmark 的任务、排行榜与 harness 会变化；本章介绍评测意图，
任何分数都必须从官方榜单/论文读取并附模型、harness、预算和日期。


31.1 五大评测对比
━━━━━━━━━━━━━━━━

┌────────────────┬──────────────┬──────────────────────┬──────────┐
│ 评测名称        │  评测什么     │       特点            │  难度     │
├────────────────┼──────────────┼──────────────────────┼──────────┤
│ GAIA           │ 通用助理     │ 多步推理+工具使用       │ ⭐⭐⭐    │
│ AgentBench     │ 多环境 Agent  │ 8 个真实环境            │ ⭐⭐⭐⭐   │
│ WebArena       │ Web 操作     │ 真实网站交互             │ ⭐⭐⭐⭐   │
│ tau-bench      │ 工具使用精度  │ 函数参数正确率            │ ⭐⭐      │
│ SWE-bench      │ 代码修复     │ 真实 GitHub Issue        │ ⭐⭐⭐⭐⭐  │
└────────────────┴──────────────┴──────────────────────┴──────────┘


31.2 GAIA —— 通用 AI 助手评测
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GAIA 是通用助理能力的一种公开参照，不应单独代表生产 Agent 质量。

评测什么：衡量 Agent 作为「通用助手」的能力——能否像真人助理一样，
完成需要多步推理 + 工具使用 + Web 浏览的复合任务。

经典例题：
  "2023年诺贝尔物理学奖得主的出生城市，在2023年的人口是多少？"

看起来简单？不——Agent 需要：
  步骤1: 搜索诺贝尔奖 → 找到获奖者姓名
  步骤2: 搜索该人的出生城市
  步骤3: 搜索该城市的人口数据
  步骤4: 组织回答

每一步都可能出错：搜索错了人名 → 整条链全废。
这就是 GAIA 的核心价值：测的是「信息检索的链式可靠性」。

GAIA 分三级（按人类完成时间）：
  Level 1: 简单（人类平均 1 分钟完成）—— 1-2 步推理
  Level 2: 中等（人类平均 3 分钟）—— 3-5 步推理 + 跨数据源
  Level 3: 困难（人类平均 10 分钟）—— 5+ 步 + 需要推理能力

使用时记录：GAIA 版本、数据访问方式、搜索工具、模型快照、重试/并发策略和
最大预算；避免把不同 harness 的榜单数字直接归因于模型本身。


31.3 AgentBench —— 8 环境综合评测
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

评测 Agent 在 8 个不同环境中的表现：
  OS / Web / Database / Code / Game / Knowledge / Card / Lateral

目标：测量 Agent 的「环境适应性」。


31.4 tau-bench —— 工具使用精度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

专门评测 Agent 的 Function Calling 准确性。
核心指标：工具选择的正确率 + 参数正确率。

报告工具评测时拆分：工具选择、参数 schema、业务约束、多轮一致性、拒绝和
副作用安全。不同模型/领域的数字不能脱离数据集复用。

SWE-bench 特别提示（2026）：Verified 已被指出存在污染和测试缺陷；对 Pro 的
进一步审查也发现任务质量问题。公开分数必须配合任务审计和私有未污染回归集。


31.5 自建评测集方法论
━━━━━━━━━━━━━━━━━━━━

四步构建法：

  1. 收集典型场景（20-50 个）
     - 高频业务场景 + 边界条件 + Bad Case 回归

  2. 定义评测维度
     - 任务完成率 / 工具选择正确率 / 响应时间 / Token 消耗

  3. 自动化评测
     - 规则检查（精确匹配/关键词）
     - LLM-as-Judge（语义质量评分）

  4. 持续迭代
     - 每次模型/prompt/tool/schema/harness 变更跑哨兵集，按风险定期跑全量
     - Bad Case → 改进 prompt → 重新评测
     - 非确定性路径运行多次，报告均值、方差和置信区间
"""

import json
import time
import hashlib
from typing import Optional


class AgentTestCase:
    """Agent 评测用例。"""

    def __init__(self, case_id: str, input_text: str,
                 expected_tools: list[str],
                 expected_keywords: list[str],
                 difficulty: str = "medium"):
        self.case_id = case_id
        self.input_text = input_text
        self.expected_tools = expected_tools
        self.expected_keywords = expected_keywords
        self.difficulty = difficulty


class SimpleAgentBench:
    """自建评测框架 —— 模拟 tau-bench 风格的工具调用评测。"""

    def __init__(self):
        self.cases = []
        self.results = []

    def add_case(self, case: AgentTestCase):
        self.cases.append(case)

    def evaluate(self, agent_func) -> dict:
        """运行评测。

        Args:
            agent_func: 返回 (tool_calls_list, answer_text) 的函数。

        Returns:
            评测报告。
        """
        for case in self.cases:
            start = time.time()
            try:
                tool_calls, answer = agent_func(case.input_text)
                success = True
            except Exception as e:
                tool_calls = []
                answer = str(e)
                success = False

            elapsed = time.time() - start

            # 工具选择评分
            tools_called = [t.get("name", "") for t in tool_calls]
            tool_score = len(set(tools_called) & set(case.expected_tools))
            tool_score /= max(len(case.expected_tools), 1)

            # 关键词评分
            kw_score = sum(1 for kw in case.expected_keywords
                          if kw.lower() in answer.lower())
            kw_score /= max(len(case.expected_keywords), 1)

            overall = (tool_score * 0.5 + kw_score * 0.5)

            self.results.append({
                "case_id": case.case_id,
                "difficulty": case.difficulty,
                "overall": round(overall, 2),
                "tool_score": round(tool_score, 2),
                "kw_score": round(kw_score, 2),
                "elapsed": round(elapsed, 2),
                "success": success,
            })

        return self._report()

    def _report(self) -> dict:
        if not self.results:
            return {"error": "无评测结果"}

        overall = sum(r["overall"] for r in self.results) / len(self.results)
        passed = sum(1 for r in self.results if r["overall"] >= 0.7)

        return {
            "total_cases": len(self.results),
            "overall_score": round(overall, 2),
            "pass_rate": f"{passed}/{len(self.results)} ({passed/len(self.results):.0%})",
            "details": self.results,
        }


def demo_benchmarks():
    print("=" * 60)
    print("  Agent 评测体系演示")
    print("=" * 60)

    bench = SimpleAgentBench()
    bench.add_case(AgentTestCase("c1", "北京天气怎么样？",
                                 ["get_weather"], ["天气", "度", "晴天"],
                                 "easy"))
    bench.add_case(AgentTestCase("c2", "搜索AI Agent并计算3+5",
                                 ["search", "calculate"], ["结果", "8"],
                                 "medium"))
    bench.add_case(AgentTestCase("c3", "分析销售额趋势并生成报告",
                                 ["query_db", "generate_chart"], ["趋势", "报告"],
                                 "hard"))

    def mock_agent(query: str):
        if "天气" in query:
            return ([{"name": "get_weather", "args": {}}],
                    "今天晴天 25°C，湿度 40%")
        if "计算" in query:
            return ([{"name": "search", "args": {}}, {"name": "calculate", "args": {}}],
                    "搜索结果：AI Agent 是... 计算结果：8")
        if "趋势" in query:
            return ([{"name": "query_db", "args": {}},
                     {"name": "generate_chart", "args": {}}],
                    "销售额呈上升趋势，详情见报告")
        return ([], "不明白您的问题")

    report = bench.evaluate(mock_agent)
    print(f"  总用例: {report['total_cases']}")
    print(f"  总分: {report['overall_score']}")
    print(f"  通过率: {report['pass_rate']}")
    for d in report["details"]:
        icon = "✅" if d["overall"] >= 0.7 else "❌"
        print(f"  {icon} {d['case_id']} ({d['difficulty']}): "
              f"overall={d['overall']} tool={d['tool_score']} "
              f"kw={d['kw_score']}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第31章：Agent 评测体系深度                             ║")
    print("║  GAIA · AgentBench · WebArena · tau-bench · SWE-bench║")
    print("╚══════════════════════════════════════════════════════╝")
    demo_benchmarks()
    print("\n▶ 五大评测对比")
    print("-" * 50)
    for name, purpose, level in [
        ("GAIA", "通用助理 多步推理+工具", "⭐⭐⭐"),
        ("AgentBench", "8环境综合适应性", "⭐⭐⭐⭐"),
        ("WebArena", "真实 Web 操作", "⭐⭐⭐⭐"),
        ("tau-bench", "工具调用精度", "⭐⭐"),
        ("SWE-bench", "代码修复", "⭐⭐⭐⭐⭐"),
    ]:
        print(f"  {name:14s} → {purpose:30s} 难度{level}")
    print("\n✅ 第31章完成！")
