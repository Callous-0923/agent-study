"""
第26章：模型路由与多 LLM 策略 —— 成本砍半的秘密
==================================================

📌 本章目标：
  1. 理解模型路由的 4 种核心策略
  2. 掌握成本-质量-延迟三元权衡
  3. 实现 Cascade Router 和 Semantic Router
  4. 学会用降级重试降低高风险查询成本

📌 面试高频点：
  - 「怎么降低 Agent 的 LLM 成本？」
  - 「什么时候该用大模型，什么时候该用小模型？」
  - 「Cascade Routing 和 Semantic Routing 的区别？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
综合 2025 RouterBench + 业界实践
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


26.1 为什么需要模型路由？
━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：
  所有请求都用 GPT-4o → 成本爆炸
  所有请求都用 GPT-4o-mini → 复杂任务质量差

类比：
  你不会用跑车去菜市场，也不会用自行车去跑高速
  不同任务需要不同的「交通工具」

模型路由要解决的问题：
  简单任务 → 便宜小模型（成本 $0.02/M token）
  复杂任务 → 强大模型（成本 $2.5/M token）
  → 保持质量的同时，降低 50-80% 成本

关键数据（业界实践）：
  智能客服场景：80% 简单查询 + 20% 复杂查询
  → 全用大模型：$1000/天
  → 路由策略：$200/天（降低 80%）
  → 质量损失：< 2%


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
  │ 延迟(额外)    │ <10ms   │ 可能+延迟 │ <50ms   │ <100ms  │
  │ 成本节省      │ ~50%    │ ~60%     │ ~70%    │ ~80%    │
  │ 质量保证      │ 较好     │ 最好      │ 较好     │ 可配置    │
  │ 适合场景      │ 通用     │ 安全第一   │ 多领域    │ 成本敏感   │
  └──────────────┴──────────┴──────────┴──────────┴──────────┘
"""

import time
import re
import json
import hashlib
from typing import Optional
from dataclasses import dataclass, field


class ModelRouter:
    """模型路由器 —— 成本优化 4 种策略的完整实现。"""

    # 模型配置（成本、能力、延迟）
    MODELS = {
        "gpt-4o": {
            "cost_per_1k_input": 0.0025,
            "cost_per_1k_output": 0.01,
            "capability_score": 10,
            "latency_estimate_ms": 2500,
        },
        "gpt-4o-mini": {
            "cost_per_1k_input": 0.00015,
            "cost_per_1k_output": 0.0006,
            "capability_score": 6,
            "latency_estimate_ms": 800,
        },
        "claude-haiku": {
            "cost_per_1k_input": 0.00025,
            "cost_per_1k_output": 0.00125,
            "capability_score": 5,
            "latency_estimate_ms": 600,
        },
        "claude-sonnet": {
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015,
            "capability_score": 9,
            "latency_estimate_ms": 1500,
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
            "model": "gpt-4o" if complexity >= threshold else "gpt-4o-mini",
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
        total_cost = 0

        # 第1步：尝试小模型
        cheap_confidence = cls._simulate_confidence(query, "gpt-4o-mini")
        step1_cost = cls._estimate_cost(query, "gpt-4o-mini")
        cascade.append({
            "step": 1, "model": "gpt-4o-mini",
            "confidence": round(cheap_confidence, 2),
            "cost": round(step1_cost, 6),
        })

        if cheap_confidence >= confidence_threshold:
            # 小模型足够好 → 完成
            return {
                "strategy": "cascade",
                "escalated": False,
                "final_model": "gpt-4o-mini",
                "total_cost": round(step1_cost, 6),
                "cascade": cascade,
            }

        # 第2步：升级大模型
        step2_cost = cls._estimate_cost(query, "gpt-4o")
        cascade.append({
            "step": 2, "model": "gpt-4o",
            "confidence": 0.95,
            "cost": round(step2_cost, 6),
        })

        return {
            "strategy": "cascade",
            "escalated": True,
            "final_model": "gpt-4o",
            "total_cost": round(step1_cost + step2_cost, 6),
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
                "model": "gpt-4o",
                "reason": "代码生成需要高能力模型",
            },
            "math": {
                "keywords": ["计算", "数学", "公式", "等于", "多少",
                            "calculate", "equation"],
                "model": "gpt-4o",
                "reason": "数学推理需要强推理能力",
            },
            "simple_qa": {
                "keywords": ["是什么", "多少钱", "时间", "在哪",
                            "怎么", "what is", "how to"],
                "model": "gpt-4o-mini",
                "reason": "简单问答小模型足够",
            },
            "writing": {
                "keywords": ["写一篇", "总结", "翻译", "润色",
                            "write", "summary", "translate"],
                "model": "claude-sonnet",
                "reason": "写作任务 Claude 表现更好",
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
                         budget: float = 0.001) -> dict:
        """策略 4：成本感知路由。

        在预算约束下选择能力最强的模型。
        """
        complexity = cls._estimate_complexity(query)
        best_model = None
        best_capability = 0

        for name, config in cls.MODELS.items():
            est_cost = cls._estimate_cost(query, name)
            if est_cost <= budget and config["capability_score"] > best_capability:
                best_model = name
                best_capability = config["capability_score"]

        return {
            "strategy": "cost_aware",
            "budget": budget,
            "model": best_model or "gpt-4o-mini",
            "est_cost": round(cls._estimate_cost(query, best_model or "gpt-4o-mini"), 6),
            "within_budget": (cls._estimate_cost(query, best_model or "gpt-4o-mini") <= budget),
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
        base = cls.MODELS[model]["capability_score"] / 10
        complexity = cls._estimate_complexity(query)
        confidence = base - complexity * 0.5
        return max(0.1, min(1.0, confidence))

    @classmethod
    def _estimate_cost(cls, query: str, model: str) -> float:
        """估算单次查询成本。"""
        cfg = cls.MODELS[model]
        input_tokens = len(query) // 4
        output_tokens = 200
        return (input_tokens / 1000 * cfg["cost_per_1k_input"] +
                output_tokens / 1000 * cfg["cost_per_1k_output"])


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
              f"| ${r2['total_cost']:.6f})")

        # 策略3
        r3 = ModelRouter.semantic_route(q)
        print(f"  Semantic:  {r3['model']:15s} "
              f"(领域:{r3['matched_domain']})")

        # 策略4
        r4 = ModelRouter.cost_aware_route(q, budget=0.0005)
        icon = "✅" if r4["within_budget"] else "❌ 超预算"
        print(f"  CostAware: {r4['model']:15s} "
              f"(${r4['est_cost']:.6f} {icon})")

    # 成本对比
    print(f"\n{'='*60}")
    print(f"  成本对比实验")
    print(f"{'='*60}")
    print(f"  场景: 100次查询(80简单+20复杂)")
    simple_q = "什么是Python？" * 80
    complex_q = "分析AI Agent的架构设计原理" * 20
    queries_100 = [simple_q] * 80 + [complex_q] * 20

    all_big = sum(ModelRouter._estimate_cost(q, "gpt-4o") for q in queries_100)
    routed = 0
    for q in queries_100:
        r = ModelRouter.threshold_route(q)
        routed += ModelRouter._estimate_cost(q, r["model"])

    print(f"  全用大模型:   ${all_big:.4f}")
    print(f"  路由策略:     ${routed:.4f}")
    print(f"  节省:         {((all_big - routed) / all_big) * 100:.0f}%")


"""
26.3 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. 模型路由 = 简单任务用小模型 + 复杂任务用大模型
2. 4 种策略各有适用场景
3. 级联路由提供质量保证（不行自动升级）
4. 合理路由可降低 50-80% 成本

面试速记：
  「怎么降低 LLM 成本？」
  → 模型路由：80% 简单用 mini，20% 复杂用大模型
  → 级联策略保证质量：小模型先试，不行再升级
  → 结果：成本降 60%+，质量损失 < 2%
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
        "Cascade: 小模型先试 → 不行升级（质量保证）",
        "Semantic: 按领域匹配最佳模型（最精准）",
        "CostAware: 预算约束下最大化能力（最可控）",
    ]:
        print(f"  {item}")
    print("\n✅ 第26章完成！")
