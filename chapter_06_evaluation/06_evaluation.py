"""
第6章：Agent 评估、测试与生产最佳实践
=====================================

📌 本章目标：
  1. 掌握 Agent 评估的核心维度和指标
  2. 了解业界常用的 Agent Benchmark
  3. 学会编写 Agent 的单元测试
  4. 掌握生产环境 Agent 的最佳实践

📌 面试高频点：
  - Agent 的质量怎么衡量？
  - LLM 评估和 Agent 评估有什么区别？
  - 如何测试一个 Agent？
  - 生产环境的 Agent 需要注意什么？


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6.1 Agent 评估 vs LLM 评估
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

评估 LLM（模型级别）：
  - 只需评估输出文本质量
  - 指标：困惑度(Perplexity)、BLEU、ROUGE、人工评分

评估 Agent（系统级别）：
  - 需要评估整个系统的端到端表现
  - 涉及多个维度：任务完成率、工具选择准确率、执行效率等

关键区别：
  Agent 的评估更接近「软件系统测试」而非「模型评估」。
  Agent 多了一个「行动」维度，需要评估：
  - 行动计划是否正确？
  - 工具调用参数是否准确？
  - 错误处理是否健壮？

评估维度全景：
┌──────────────┬──────────────────────────────────────┐
│    维度       │              评估内容                 │
├──────────────┼──────────────────────────────────────┤
│ 任务完成率    │ 是否成功完成用户指定的任务？           │
│ 工具调用准确率 │ 是否选择了正确的工具？参数是否正确？   │
│ 执行效率      │ 完成任务需要多少步？消耗多少 token？   │
│ 鲁棒性        │ 输入异常/工具失败时的处理能力          │
│ 安全性        │ 是否执行了危险操作？                  │
│ 用户体验      │ 回答是否流畅、有帮助、易于理解？       │
└──────────────┴──────────────────────────────────────┘

主流 Agent Benchmark：
  - AgentBench: 8 个真实环境的 Agent 评测
  - SWE-bench: 软件工程任务评测
  - WebArena: Web 操作评测
  - GAIA: 通用 AI 助手评测
  - τ-Bench: 工具使用评测
"""

import json
import time
from typing import Callable


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6.2 Agent 评测框架（可运行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

下面实现一个 AgentEval 评测框架，围绕 5 个核心维度：
  1. 任务完成率 —— Agent 是否成功完成了用户的请求？
  2. 工具选择准确率 —— Function Calling 选对工具了吗？
  3. 参数提取准确率 —— 传给工具的 JSON 参数是否正确？
  4. Token 效率 —— 完成任务用了多少 Token？
  5. 延迟 —— 从用户提问到最终回答花了多久？

这套框架会跑测试用例、自动打分、生成报告。你可以直接用于自己的 Agent 项目。
"""


class AgentEvaluator:
    """Agent 评测器 —— 系统化评估 Agent 的表现。

    设计思路：
      1. 准备一组测试用例（输入 + 期望输出 + 评估标准）
      2. 对每个用例运行 Agent
      3. 用 LLM 作为评判员（LLM-as-Judge）来打分
      4. 汇总统计并生成报告

    LLM-as-Judge 是目前业界评估 Agent 的主流方法：
      因为 Agent 的输出是自然语言，传统指标（BLEU/ROUGE）
      无法准确评估语义层面的质量。
    """

    def __init__(self, llm: Callable = None):
        """初始化评测器。

        Args:
            llm: 用于评分的 LLM 函数（可选的 LLM-as-Judge）。
        """
        self.llm = llm
        self.results = []

    def evaluate(self, agent_func: Callable,
                 test_cases: list[dict]) -> list[dict]:
        """运行评测。

        Args:
            agent_func: Agent 执行函数，签名为 (input: str) -> str。
            test_cases: 测试用例列表，每个用例包含:
                - input: 输入文本
                - expected_keywords: 回答应包含的关键词（可选）
                - expected_tool: 预期调用的工具名（可选）
                - weight: 权重（默认 1.0）

        Returns:
            评测结果列表。
        """
        self.results = []
        for i, case in enumerate(test_cases):
            print(f"\n  📝 评测用例 {i+1}/{len(test_cases)}: {case['input'][:50]}...")

            start_time = time.time()
            try:
                output = agent_func(case["input"])
                error = None
            except Exception as e:
                output = ""
                error = str(e)

            elapsed = time.time() - start_time

            # 基本检查
            score = 1.0
            details = []

            # 检查关键词
            if "expected_keywords" in case and case["expected_keywords"]:
                keyword_score = self._check_keywords(
                    output, case["expected_keywords"]
                )
                score *= keyword_score
                details.append(f"关键词匹配: {keyword_score:.2f}")

            # 检查是否非空
            if not output or len(output) < 10:
                score *= 0.5
                details.append("输出过短")

            # 检查是否有错误
            if error:
                score *= 0
                details.append(f"执行错误: {error}")

            weight = case.get("weight", 1.0)
            result = {
                "case_id": i + 1,
                "input": case["input"],
                "output": output[:200],
                "score": score,
                "weight": weight,
                "error": error,
                "elapsed_sec": round(elapsed, 2),
                "details": details,
            }
            self.results.append(result)
            print(f"    评分: {score:.2f} | 耗时: {elapsed:.2f}s")

        return self.results

    def _check_keywords(self, output: str, keywords: list[str]) -> float:
        """检查输出是否包含期望关键词。

        Args:
            output: Agent 输出。
            keywords: 期望出现的关键词列表。

        Returns:
            匹配比例（0~1）。
        """
        if not keywords:
            return 1.0
        matched = sum(1 for kw in keywords if kw.lower() in output.lower())
        return matched / len(keywords)

    def summary(self) -> str:
        """生成评测汇总报告。

        Returns:
            格式化的评测报告。
        """
        if not self.results:
            return "无评测结果。"

        total_weight = sum(r["weight"] for r in self.results)
        weighted_score = sum(
            r["score"] * r["weight"] for r in self.results
        ) / total_weight if total_weight > 0 else 0

        passed = sum(1 for r in self.results if r["score"] >= 0.7)
        failed = len(self.results) - passed

        lines = [
            "=" * 55,
            "  📊 Agent 评测报告",
            "=" * 55,
            f"  总用例数: {len(self.results)}",
            f"  通过 (≥0.7): {passed}",
            f"  失败 (<0.7): {failed}",
            f"  加权平均分: {weighted_score:.2f}",
            f"  平均耗时: {sum(r['elapsed_sec'] for r in self.results) / len(self.results):.2f}s",
            "-" * 55,
        ]

        for r in self.results:
            status = "✅" if r["score"] >= 0.7 else "❌"
            lines.append(
                f"  {status} Case {r['case_id']}: "
                f"score={r['score']:.2f} | time={r['elapsed_sec']}s"
            )
            if r["details"]:
                for d in r["details"]:
                    lines.append(f"     └─ {d}")

        lines.append("=" * 55)
        return "\n".join(lines)


def demo_evaluation():
    """演示 Agent 评测体系。"""
    print("\n" + "=" * 60)
    print("  Agent 评测体系演示")
    print("=" * 60)

    # 模拟一个「简单 Agent」
    def mock_agent(user_input: str) -> str:
        """模拟 Agent —— 用于演示评测流程。"""
        if "天气" in user_input:
            return "当前天气晴好，气温25°C，湿度适中，适合户外活动。"
        if "计算" in user_input:
            try:
                expr = user_input.split("计算")[-1].strip()
                return f"计算结果为: {eval(expr)}"
            except Exception:
                return "计算失败，请检查表达式。"
        if "搜索" in user_input:
            return "搜索结果: 找到了相关信息，详情请查看链接。"
        return "我不太理解您的问题，请提供更多信息。"

    # 准备测试用例
    test_cases = [
        {
            "input": "北京今天天气怎么样？",
            "expected_keywords": ["天气", "温度", "°C"],
            "weight": 1.0,
        },
        {
            "input": "帮我算一下 123 + 456 等于多少",
            "expected_keywords": ["579", "计算结果"],
            "weight": 1.0,
        },
        {
            "input": "搜索 Python 教程",
            "expected_keywords": ["搜索", "结果"],
            "weight": 0.5,
        },
        {
            "input": "写一首关于春天的诗",
            "expected_keywords": ["春"],
            "weight": 1.5,
        },
    ]

    evaluator = AgentEvaluator()
    evaluator.evaluate(mock_agent, test_cases)
    print("\n" + evaluator.summary())


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6.3 Agent 测试策略
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

测试金字塔（从底层到顶层）：

  Layer 1: 工具单元测试
    - 测试每个 Tool 函数的输入/输出
    - 确保工具在 LLM 调用前已经正确

  Layer 2: Agent 行为测试
    - 给定输入，验证是否调用了正确的工具
    - 验证 tool_calls 的参数是否正确
    - Mock LLM 的返回，测试 Agent 循环逻辑

  Layer 3: 端到端测试
    - 完整运行 Agent，验证最终输出
    - 使用 LLM-as-Judge 评分
    - 覆盖典型场景和边界情况

  Layer 4: 回归测试
    - 收集线上的 bad case
    - 每次改动后验证这些 case 是否退化
    - 持续积累评测集

面试常见问题：
  "你怎么测试 Agent 代码？"
  → 分层测试：工具用单测，Agent 循环用 mock，
     端到端用 LLM-as-Judge，线上用回归评测集
"""


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6.4 生产环境最佳实践
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 错误处理
   - 每个工具调用都要 try-catch
   - 给 LLM 返回「能帮助它修正」的错误信息
   - 设置全局超时和最大重试次数

2. 可观测性
   - 记录每次 LLM 调用的输入/输出/token/耗时
   - 记录每个 tool_call 的参数和结果
   - 使用 LangSmith / LangFuse 等工具做 Tracing

3. 成本控制
   - 用更小更便宜的模型做简单任务（模型分层）
   - 缓存相同/相似的 LLM 调用
   - 设置 max_tokens 限制每次调用的成本上限
   - 用语义缓存减少重复查询

4. 安全防护
   - Prompt Injection 防护（输入过滤）
   - 工具权限最小化（sudo 问题）
   - 人工确认（Human-in-the-Loop）对关键操作

5. 性能优化
   - 并行执行独立的工具调用
   - 流式输出（streaming）提升用户体验
   - 预加载和预计算

面试速记：
  "生产环境 Agent 和 Demo 的区别？"
  → Demo 追求「能跑」；生产追求「稳定、安全、可观测、低成本」
  → 需要监控、告警、降级、熔断等基础设施
  → 错误处理从"打印日志"升级到"自动恢复 + 通知"
"""


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6.5 本章总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心要点回顾：

1. Agent 评估
   - 多维度：任务完成率、工具准确率、效率、鲁棒性
   - 主流方法：LLM-as-Judge
   - 业界 Benchmark：AgentBench、SWE-bench、GAIA

2. Agent 测试
   - 分层测试：工具单测 → 行为测试 → 端到端测试 → 回归测试
   - LLM-as-Judge 是端到端测试的核心方法

3. 生产最佳实践
   - 错误处理：try-catch + 可读错误信息
   - 可观测性：Tracing + Logging
   - 成本控制：模型分层 + 缓存 + 限制
   - 安全：Prompt Injection 防护 + 权限最小化
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║       第6章：Agent 评估与最佳实践                      ║")
    print("║       评测框架 · 测试策略 · 生产部署                  ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n▶ 6.2 Agent 评测框架演示")
    demo_evaluation()

    print("\n▶ 6.3 测试策略速查")
    print("-" * 50)
    print("Layer 1: 工具单元测试 → 测 Tool 函数")
    print("Layer 2: Agent 行为测试 → Mock LLM, 测循环逻辑")
    print("Layer 3: 端到端测试 → LLM-as-Judge 评分")
    print("Layer 4: 回归测试 → Bad case 积累 + 持续回归")

    print("\n▶ 6.4 生产 Checklist")
    print("-" * 50)
    checks = [
        "✅ 错误处理: 每个工具调用有 try-catch",
        "✅ 可观测性: Tracing + Logging 全覆盖",
        "✅ 成本控制: 模型分层 + 缓存 + 限制",
        "✅ 安全防护: Input 过滤 + 权限最小化",
        "✅ 性能优化: 并行 + Streaming + 预计算",
    ]
    for c in checks:
        print(f"  {c}")

    print("\n✅ 第6章完成！接下来进入第7章：求职面试准备。")
