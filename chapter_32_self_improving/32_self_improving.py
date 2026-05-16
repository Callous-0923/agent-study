"""
第32章：Self-Improving Agent —— 从错误中进化的 Agent
======================================================

📌 本章目标：
  1. 理解 Self-Improving Agent 的闭环架构
  2. 掌握 Bad Case 收集 → 自动改 Prompt → 评测验证的流程
  3. 实现简单的 Self-Improvement 循环
  4. 了解 Reflection 和 Self-Improve 的区别

📌 面试高频点：
  - 「Agent 怎么从错误中学习？」
  - 「Self-Improving 和 Fine-tuning 的区别？」
  - 「你怎么保证改进不会引入新问题？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
参考：Reflexion (Shinn et al. 2023) + DSPy Optimizer 实践
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


32.1 什么是 Self-Improving Agent？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

把 Agent 想象成实习生：传统 Agent 是那种「永远不改进」的实习生——
犯了错下次还犯，Prompt 一成不变。Self-Improving Agent 则是那种
「会总结教训」的实习生——每次出错后自己分析原因，优化工作方法。

技术上，Self-Improving 指的是 Agent 能自动从交互数据中识别问题、
调整自己的行为（通常通过修改 Prompt 或路由策略）、再验证改进效果
——全程不需要人类介入。

传统 Agent：固定 Prompt → 固定逻辑 → 永远不进步
Self-Improving：执行 → 出错 → 分析原因 → 自动修改 Prompt → 验证改进

为什么这个能力对面试官来说如此重要？因为它回答了 Agent 领域最大的
质疑：「Agent 上线后怎么持续变好？」——Self-Improving 就是答案。

不过要澄清一个常见混淆：Self-Improving 不是 Fine-tuning。
Fine-tuning 是改模型权重，需要 GPU；Self-Improving 是改 Prompt 或
路由规则，只需要让 LLM 分析 Bad Case 就能触发。成本低、可逆、
可以频繁运行。

核心循环：
  ┌─────────────────────────────────────────┐
  │                                           │
  │  执行任务 → 收集结果 → 评测打分            │
  │      ↑                      ↓             │
  │      │              低分案例（Bad Case）    │
  │      │                      ↓             │
  │      └──── 改进 Prompt ← 分析原因          │
  │                                           │
  └─────────────────────────────────────────┘


32.2 Self-Improving vs Reflection vs Fine-tuning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────┬────────────────────┬─────────────────────┐
│ 方式             │       机制          │        成本          │
├────────────────┼────────────────────┼─────────────────────┤
│ Reflection     │ 单次任务内自我审查    │ 低（多一次 LLM 调用）  │
│ Self-Improving │ 跨任务的持续改进       │ 中（定期跑优化流程）    │
│ Fine-tuning    │ 重新训练模型权重       │ 高（GPU + 数据标注）   │
└────────────────┴────────────────────┴─────────────────────┘

Reflection（Ch3/Ch19）：一次任务内的自我批评
Self-Improving（本章）：横跨多次任务的持续改进
"""

import json
import time
import hashlib
from collections import defaultdict


class SelfImprovingAgent:
    """自改进 Agent —— Bad Case 驱动的 Prompt 优化。"""

    def __init__(self, base_prompt: str = "",
                 metric_func=None):
        self.base_prompt = base_prompt
        self.improved_prompt = base_prompt
        self.metric = metric_func or (lambda p, e: 1.0 if p == e else 0.0)
        self.bad_cases = []
        self.improvement_log = []
        self.version = 1

    def execute(self, task: str) -> str:
        """执行任务（模拟 LLM 调用）。"""
        return f"[v{self.version}] 对 '{task[:30]}...' 的回答"

    def collect_result(self, task: str, expected: str):
        """收集一次执行结果，记录 Bad Case。

        Args:
            task: 任务输入。
            expected: 期望输出。
        """
        actual = self.execute(task)
        score = self.metric(actual, expected)

        if score < 0.7:
            self.bad_cases.append({
                "task": task,
                "expected": expected,
                "actual": actual,
                "score": score,
                "timestamp": time.time(),
            })

        return score

    def improve(self) -> dict:
        """基于 Bad Case 自动改进 Prompt。

        Returns:
            改进报告。
        """
        if len(self.bad_cases) < 3:
            return {"improved": False,
                    "reason": f"Bad Case 不足 ({len(self.bad_cases)} < 3)"}

        # 分析 Bad Case 模式
        patterns = self._analyze_bad_cases()

        # 生成改进后的 Prompt
        new_prompt = self._generate_improved_prompt(patterns)

        improvement = {
            "version": self.version,
            "patterns_found": patterns,
            "old_prompt_len": len(self.improved_prompt),
            "new_prompt_len": len(new_prompt),
            "bad_case_count": len(self.bad_cases),
        }

        # 应用改进
        self.improved_prompt = new_prompt
        self.version += 1
        self.bad_cases = []
        self.improvement_log.append(improvement)

        return {"improved": True, **improvement}

    def _analyze_bad_cases(self) -> list[str]:
        """分析 Bad Case 的共性模式。"""
        patterns = []
        tasks = [bc["task"] for bc in self.bad_cases]

        # 模式1：如果多个 Bad Case 包含相同关键词
        common_words = set(tasks[0].split())
        for t in tasks[1:]:
            common_words &= set(t.split())

        if common_words:
            patterns.append(f"共同关键词: {', '.join(list(common_words)[:5])}")

        # 模式2：评分分布
        avg_score = sum(bc["score"] for bc in self.bad_cases) / len(self.bad_cases)
        patterns.append(f"平均评分: {avg_score:.2f}")

        return patterns

    def _generate_improved_prompt(self, patterns: list) -> str:
        """生成改进后的 Prompt（模拟 LLM 自动优化）。

        真实实现中调用 LLM：
          "以下是我的 Bad Case 模式: {patterns}
           当前 Prompt: {current_prompt}
           请生成改进后的 Prompt，要求更精确、容错性更好"
        """
        additions = [
            f"\n\n[自改进 v{self.version}] 基于 {len(self.bad_cases)} 个 Bad Case:",
            "- " + "\n- ".join(patterns),
            "- 请特别注意以上模式，给出更精确的回答。",
        ]
        return self.improved_prompt + "\n".join(additions)


def demo_self_improving():
    print("=" * 60)
    print("  Self-Improving Agent 演示")
    print("=" * 60)

    def exact_metric(pred: str, exp: str) -> float:
        # 简化版：包含期望关键词即得分
        return sum(1 for w in exp.split() if w in pred) / max(len(exp.split()), 1)

    agent = SelfImprovingAgent(
        base_prompt="你是一个 AI 助手，请回答问题。",
        metric_func=exact_metric,
    )

    # 模拟多次交互
    test_cases = [
        ("北京天气", "晴天 25°C 湿度 40%"),
        ("上海天气", "多云 28°C 湿度 65%"),
        ("深圳天气", "阵雨 30°C 湿度 80%"),
        ("成都天气", "阴天 22°C 湿度 55%"),
        ("杭州天气", "小雨 20°C 湿度 70%"),
    ]

    print(f"\n  📝 初始 Prompt({len(agent.improved_prompt)}字符): "
          f"{agent.improved_prompt[:50]}...")

    for task, expected in test_cases:
        score = agent.collect_result(task, expected)
        print(f"    「{task}」→ 评分: {score:.2f}")

    print(f"\n  📊 Bad Case 数: {len(agent.bad_cases)}")

    result = agent.improve()
    if result["improved"]:
        print(f"  ✅ 第 {result['version']} 次改进完成!")
        print(f"  发现的模式: {result['patterns_found']}")
        print(f"  Prompt 长度: {result['old_prompt_len']} → {result['new_prompt_len']}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第32章：Self-Improving Agent                           ║")
    print("║  Bad Case 收集 · 自动改 Prompt · 评测验证             ║")
    print("╚══════════════════════════════════════════════════════╝")
    demo_self_improving()
    print("\n▶ Self-Improving vs Reflection vs FT")
    print("-" * 50)
    for name, desc in [
        ("Reflection", "单次任务内的自我审查"),
        ("Self-Improving", "跨任务持续改进 Prompt"),
        ("Fine-tuning", "GPU训练 + 标注数据"),
    ]:
        print(f"  {name:18s} → {desc}")
    print("\n✅ 第32章完成！")
