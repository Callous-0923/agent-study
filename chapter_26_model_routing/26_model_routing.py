"""
第26章：模型路由与多 LLM 策略 —— 用评测约束质量与预算
==================================================

内容核对：2026-08-01
说明：标注为模拟的实现与数值用于讲解概念，不代表真实 SDK、协议或基准结果。

📌 本章目标：
  1. 理解模型路由的 4 种核心策略
  2. 掌握成本-质量-延迟三元权衡
  3. 实现 Cascade Router 和 Semantic Router
  4. 学会用降级重试降低高风险查询成本

📌 面试高频点：
  - 「怎么降低 Agent 的 LLM 成本？」
  - 「什么时候该用大模型，什么时候该用小模型？」
  - 「Cascade Routing 和 Semantic Routing 的区别？」

模型目录、价格和限额会变化。本章代码使用相对成本单位和可配置模型 ID，
不把某一天的供应商价格或 benchmark 排名固化为“业界真实数字”。


26.1 为什么需要模型路由？—— 不是「选最好的」，而是「选最合适的」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

初学者常犯的错误是所有请求都用最高能力模型，而没有先定义质量门槛。

如果只有一个模型可用，路由没有意义；有多个候选时，先在自己的任务集上测量
正确率、失败类型、延迟和实际账单，再按风险与预算分配。

类比：你不会用跑车去菜市场（油费贵、停车难），也不会用自行车去跑
高速（根本上不去）。模型路由就是给每个请求配最合适的「交通工具」。

模型路由要解决的问题：
  低风险、短回答 → 先尝试 economy / balanced 候选
  高风险、复杂推理 → high_capability，或直接转人工
  最终效果 → 只有离线回归和线上护栏均达标时才计算节省

面试官想听的不是「我们用 GPT-4o」，而是「我们根据什么策略选模型」。


26.2 四种路由策略
━━━━━━━━━━━━━━━

  ┌─────────────────────┬──────────────────────────────────┐
  │ 策略                 │            核心思想               │
  ├─────────────────────┼──────────────────────────────────┤
  │ 1. Threshold Router │ 复杂度评分 → 超过阈值用大模型      │
  │ 2. Cascade Router   │ 先用小模型 → 不行再升级            │
  │ 3. Semantic Router  │ 语义相似度 → 匹配最佳模型           │
  │ 4. Cost-Aware Router │ 成本约束下最大化质量               │
  └─────────────────────┴──────────────────────────────────┘

策略对比：
  ┌──────────────┬──────────┬──────────┬──────────┬──────────┐
  │              │Threshold │ Cascade  │ Semantic │Cost-Aware│
  ├──────────────┼──────────┼──────────┼──────────┼──────────┤
  │ 实现难度      │ ⭐       │ ⭐⭐      │ ⭐⭐⭐     │ ⭐⭐⭐     │
  │ 延迟特征      │ 单次分类  │ 可能多次调用│ 检索/分类 │ 需先估算  │
  │ 节省          │ 任务相关  │ 任务相关   │ 任务相关  │ 预算约束  │
  │ 质量保证      │ 需回归集  │ 需验证升级判据│ 需回归集 │ 需设下限  │
  │ 适合场景      │ 通用     │ 安全第一   │ 多领域    │ 成本敏感   │
  └──────────────┴──────────┴──────────┴──────────┴──────────┘
"""

import time
import re
import json
import hashlib
import os
from typing import Optional
from dataclasses import dataclass, field


class ModelRouter:
    """模型路由器 —— 成本优化 4 种策略的完整实现。"""

    # 相对单位只用于解释算法；生产中应从配置服务加载当前价格和评测结果。
    MODELS = {
        "high_capability": {
            "model_id": os.getenv("ROUTER_HIGH_MODEL", "gpt-5.6-sol"),
            "input_cost_units": 8.0,
            "output_cost_units": 10.0,
            "capability_tier": 3,
        },
        "balanced": {
            "model_id": os.getenv("ROUTER_BALANCED_MODEL", "gpt-5.6-terra"),
            "input_cost_units": 3.0,
            "output_cost_units": 4.0,
            "capability_tier": 2,
        },
        "economy": {
            "model_id": os.getenv("ROUTER_ECONOMY_MODEL", "gpt-5.6-luna"),
            "input_cost_units": 1.0,
            "output_cost_units": 1.5,
            "capability_tier": 1,
        },
    }

    @classmethod
    def threshold_route(cls, query: str,
                        threshold: float = 0.5) -> dict:
        """策略 1：复杂度阈值路由。

        最简单有效的策略：
          复杂度 < 阈值 → 小模型
          复杂度 ≥ 阈值 → 大模型
        """
        complexity = cls._estimate_complexity(query)
        return {
            "strategy": "threshold",
            "complexity": round(complexity, 2),
            "model": "high_capability" if complexity >= threshold else "balanced",
            "reason": (f"复杂度 {complexity:.2f} "
                       f"{'≥' if complexity >= threshold else '<'} "
                       f"阈值 {threshold}"),
        }

    @classmethod
    def cascade_route(cls, query: str,
                      confidence_threshold: float = 0.7) -> dict:
        """策略 2：级联路由（降级重试）。

        先用便宜模型 → 置信度不够 → 自动升级大模型。
        保证最终质量不下降。
        """
        cascade = []

        # 第1步：尝试小模型
        cheap_confidence = cls._simulate_confidence(query, "economy")
        step1_cost = cls._estimate_cost_units(query, "economy")
        cascade.append({
            "step": 1, "model": "economy",
            "confidence": round(cheap_confidence, 2),
            "cost_units": round(step1_cost, 3),
        })

        if cheap_confidence >= confidence_threshold:
            # 小模型足够好 → 完成
            return {
                "strategy": "cascade",
                "escalated": False,
                "final_model": "economy",
                "total_cost_units": round(step1_cost, 3),
                "cascade": cascade,
            }

        # 第2步：升级大模型
        step2_cost = cls._estimate_cost_units(query, "high_capability")
        cascade.append({
            "step": 2, "model": "high_capability",
            "confidence": 0.95,
            "cost_units": round(step2_cost, 3),
        })

        return {
            "strategy": "cascade",
            "escalated": True,
            "final_model": "high_capability",
            "total_cost_units": round(step1_cost + step2_cost, 3),
            "cascade": cascade,
        }

    @classmethod
    def semantic_route(cls, query: str) -> dict:
        """策略 3：语义路由。

        根据查询的语义特征匹配最合适的模型。
        不同模型擅长不同领域。
        """
        domain_examples = {
            "code": {
                "keywords": ["代码", "函数", "bug", "debug", "算法",
                            "写一个", "实现", "code", "function"],
                "model": "high_capability",
                "reason": "代码生成需要高能力模型",
            },
            "math": {
                "keywords": ["计算", "数学", "公式", "等于", "多少",
                            "calculate", "equation"],
                "model": "high_capability",
                "reason": "数学推理需要强推理能力",
            },
            "simple_qa": {
                "keywords": ["是什么", "多少钱", "时间", "在哪",
                            "怎么", "what is", "how to"],
                "model": "balanced",
                "reason": "简单问答小模型足够",
            },
            "writing": {
                "keywords": ["写一篇", "总结", "翻译", "润色",
                            "write", "summary", "translate"],
                "model": "balanced",
                "reason": "写作候选应由语言与风格评测决定",
            },
        }

        scores = {}
        for domain, config in domain_examples.items():
            score = sum(1 for kw in config["keywords"]
                       if kw.lower() in query.lower())
            scores[domain] = score

        best_domain = max(scores, key=scores.get)
        config = domain_examples[best_domain]

        return {
            "strategy": "semantic",
            "matched_domain": best_domain,
            "model": config["model"],
            "reason": config["reason"],
            "score": scores[best_domain],
        }

    @classmethod
    def cost_aware_route(cls, query: str,
                         budget_units: float = 1.0) -> dict:
        """策略 4：成本感知路由。

        在预算约束下选择能力最强的模型。
        """
        complexity = cls._estimate_complexity(query)
        best_model = None
        best_capability = 0

        for name, config in cls.MODELS.items():
            est_cost = cls._estimate_cost_units(query, name)
            if est_cost <= budget_units and config["capability_tier"] > best_capability:
                best_model = name
                best_capability = config["capability_tier"]

        selected = best_model or "economy"
        return {
            "strategy": "cost_aware",
            "budget_units": budget_units,
            "model": selected,
            "model_id": cls.MODELS[selected]["model_id"],
            "est_cost_units": round(cls._estimate_cost_units(query, selected), 3),
            "within_budget": cls._estimate_cost_units(query, selected) <= budget_units,
        }

    # ========== 辅助方法 ==========

    @staticmethod
    def _estimate_complexity(query: str) -> float:
        """估算查询复杂度（0-1）。

        规则：关键词 + 长度 + 标点综合判断。
        """
        score = 0.0
        lq = query.lower()

        # 推理关键词 → 高复杂度
        reasoning_kw = ["为什么", "如何", "分析", "解释", "原因",
                        "设计", "方案", "对比", "why", "explain",
                        "analyze", "compare"]
        score += sum(0.15 for kw in reasoning_kw if kw in lq)

        # 代码关键词 → 高复杂度
        code_kw = ["代码", "函数", "实现", "bug", "code", "function"]
        score += sum(0.15 for kw in code_kw if kw in lq)

        # 简单问答 → 低复杂度
        simple_kw = ["是什么", "多少钱", "几点", "在哪", "what is"]
        score -= sum(0.1 for kw in simple_kw if kw in lq)

        # 长度影响
        if len(query) > 200:
            score += 0.1

        return max(0.0, min(1.0, score))

    @classmethod
    def _simulate_confidence(cls, query: str, model: str) -> float:
        """模拟模型的置信度。"""
        base = cls.MODELS[model]["capability_tier"] / 3
        complexity = cls._estimate_complexity(query)
        confidence = base - complexity * 0.5
        return max(0.1, min(1.0, confidence))

    @classmethod
    def _estimate_cost_units(cls, query: str, model: str) -> float:
        """估算相对成本单位；不是美元价格。"""
        cfg = cls.MODELS[model]
        input_tokens = max(1, len(query) // 4)
        output_tokens = 200
        return (input_tokens / 1000 * cfg["input_cost_units"] +
                output_tokens / 1000 * cfg["output_cost_units"])


def demo_routing():
    """演示四种路由策略。"""
    print("=" * 60)
    print("  模型路由 4 策略演示")
    print("=" * 60)

    queries = [
        "帮我写一个快速排序的Python函数",
        "今天天气怎么样？",
        "分析一下AI Agent市场的竞争格局",
        "帮我翻译这段文字成英文",
        "计算 12345 * 67890 等于多少",
    ]

    for q in queries:
        print(f"\n  ── 查询: 「{q[:40]}...」──")

        # 策略1
        r1 = ModelRouter.threshold_route(q)
        print(f"  Threshold: {r1['model']:15s} (复杂度:{r1['complexity']})")

        # 策略2
        r2 = ModelRouter.cascade_route(q)
        print(f"  Cascade:   {r2['final_model']:15s} "
              f"({'升级' if r2['escalated'] else '未升级'} "
              f"| {r2['total_cost_units']:.3f} units)")

        # 策略3
        r3 = ModelRouter.semantic_route(q)
        print(f"  Semantic:  {r3['model']:15s} "
              f"(领域:{r3['matched_domain']})")

        # 策略4
        r4 = ModelRouter.cost_aware_route(q, budget_units=1.0)
        icon = "✅" if r4["within_budget"] else "❌ 超预算"
        print(f"  CostAware: {r4['model']:15s} "
              f"({r4['est_cost_units']:.3f} units {icon})")

    # 成本对比
    print(f"\n{'='*60}")
    print(f"  成本对比实验")
    print(f"{'='*60}")
    print(f"  教学场景: 100次查询(80简单+20复杂)，不代表真实流量分布")
    queries_100 = ["什么是Python？"] * 80 + ["分析AI Agent的架构设计原理"] * 20

    all_big = sum(ModelRouter._estimate_cost_units(q, "high_capability") for q in queries_100)
    routed = 0
    for q in queries_100:
        r = ModelRouter.threshold_route(q)
        routed += ModelRouter._estimate_cost_units(q, r["model"])

    print(f"  全用高能力模型: {all_big:.3f} relative units")
    print(f"  路由策略:       {routed:.3f} relative units")
    print(f"  模拟单位下降:   {((all_big - routed) / all_big) * 100:.0f}%")
    print("  生产结论还需同时满足质量、风险与延迟门槛。")


"""
26.3 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. 模型路由 = 简单任务用小模型 + 复杂任务用大模型
2. 4 种策略各有适用场景
3. 级联路由只在升级判据可靠且高能力模型通过回归时改善质量
4. 成本下降幅度取决于任务分布、价格、缓存和升级率

面试速记：
  「怎么降低 LLM 成本？」
  → 先构建任务分层与质量门槛，再评测 economy/balanced/high_capability
  → 级联判据用可校准 verifier，不把模型自报置信度当事实
  → 报告路由前后质量、风险、p95 延迟和实际账单
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第26章：模型路由与多 LLM 策略                         ║")
    print("║  Threshold · Cascade · Semantic · Cost-Aware        ║")
    print("╚══════════════════════════════════════════════════════╝")
    demo_routing()
    print("\n▶ 4 种路由策略速查")
    print("-" * 50)
    for item in [
        "Threshold: 复杂度评分 > 阈值 → 大模型（最简单）",
        "Cascade: 低成本候选先试 → verifier 不通过则升级",
        "Semantic: 按任务簇匹配经评测的候选模型",
        "CostAware: 在质量下限与预算约束下选择",
    ]:
        print(f"  {item}")
    print("\n✅ 第26章完成！")
