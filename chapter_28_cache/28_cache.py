"""
第28章：语义缓存与 Token 优化 —— 用正确性约束缓存收益
========================================================

内容核对：2026-08-01
说明：标注为模拟的实现与数值用于讲解概念，不代表真实 SDK、协议或基准结果。

📌 本章目标：
  1. 理解语义缓存的核心原理和三级缓存架构
  2. 掌握 Exact Match / Embedding / LLM-as-Cache 的实现
  3. 学会量化缓存收益和缓存命中率预测
  4. 掌握 Token 预算管理系统

📌 面试高频点：
  - 「Agent 的 Token 成本怎么优化？」
  - 「语义缓存怎么实现？和普通缓存有什么区别？」
  - 「缓存命中率怎么估算？实际能省多少钱？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
综合 CDN 缓存思想 + LLM 成本优化实践
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


28.1 为什么 Agent 场景特别需要缓存？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent 的 LLM 调用特征：
  1. 某些 FAQ 场景存在重复请求，但实际比例必须从脱敏日志测量
  2. 代价高：每次调用 = 金钱 + 时间
  3. 步骤多：一个 Agent 任务可能调 5-10 次 LLM

缓存收益公式：
  避免调用数 = 可缓存请求数 × 实测命中率 × 正确性通过率
  净收益 = 避免的实际账单 - embedding/存储/失效/误命中的处理成本
  延迟收益也应从线上 p50/p95/p99 测量，不引用跨系统固定数字。


28.2 三级缓存架构
━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────┐
  │                 L1: Exact Match Cache              │
  │  ─────────────────────────────────               │
  │  查询: "今天天气" → 缓存中查 "今天天气"             │
  │  命中条件: 完全相同的文本                           │
  │  命中率/延迟: 由 key、存储和流量分布实测             │
  ├─────────────────────────────────────────────────┤
  │                 L2: Semantic Cache                 │
  │  ─────────────────────────────────               │
  │  查询: "今天天气" → 缓存中查相似查询 → 命中 "今天天气怎么样"│
  │  命中条件: 阈值由标注对和风险等级校准                 │
  │  同时监控命中率、误命中率和数据新鲜度                 │
  ├─────────────────────────────────────────────────┤
  │                 L3: LLM Fallback                   │
  │  ─────────────────────────────────               │
  │  查询: 无法在前2级命中 → 调用廉价小模型               │
  │  这不是 cache hit；生成后仅在策略允许时写回 L1/L2     │
  │  质量、延迟和费用按模型与任务实测                     │
  └─────────────────────────────────────────────────┘
"""

import time
import json
import hashlib
import numpy as np
from typing import Optional
from collections import OrderedDict


class ExactMatchCache:
    """L1 缓存 —— 精确匹配。

    最简单的缓存：完全相同的输入 → 直接返回缓存结果。
    """

    def __init__(self, max_size: int = 1000):
        self._store = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> Optional[str]:
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self._store:
            self._store.move_to_end(key)  # LRU 更新
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def set(self, query: str, response: str):
        key = hashlib.md5(query.encode()).hexdigest()
        if len(self._store) >= self.max_size:
            self._store.popitem(last=False)  # 淘汰最久未用
        self._store[key] = response

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


class SemanticCache:
    """L2 缓存 —— 语义相似度匹配。

    不要求完全相同的查询，语义相似即可命中。
    """

    def __init__(self, similarity_threshold: float = 0.92,
                 max_size: int = 2000):
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.queries = []      # [(embedding, query, response)]
        self.hits = 0
        self.misses = 0

    def _embed(self, text: str, dim: int = 128) -> np.ndarray:
        """简化的向量化（实际使用 OpenAI Embeddings API）。"""
        vec = np.zeros(dim)
        for i, ch in enumerate(text):
            vec[hash(ch) % dim] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def get(self, query: str) -> Optional[str]:
        q_vec = self._embed(query)
        for emb, cached_query, response in self.queries:
            sim = float(np.dot(q_vec, emb))
            if sim >= self.threshold:
                self.hits += 1
                return response
        self.misses += 1
        return None

    def set(self, query: str, response: str):
        emb = self._embed(query)
        self.queries.append((emb, query, response))
        if len(self.queries) > self.max_size:
            self.queries = self.queries[-self.max_size:]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


class ThreeLevelCache:
    """两级缓存 + LLM fallback 的教学流水线。"""

    def __init__(self):
        self.l1 = ExactMatchCache(max_size=500)
        self.l2 = SemanticCache(similarity_threshold=0.90)
        self.stats = {"l1_hits": 0, "l2_hits": 0,
                      "llm_calls": 0, "misses": 0,
                      "avoided_llm_calls": 0}

    def query(self, query: str,
              llm_func: callable = None) -> tuple[str, int, bool]:
        """三级缓存查询。

        Returns:
            (response, cache_level, was_cached)
            cache_level: 1=L1, 2=L2, 3=L3, 0=miss
        """
        # L1: Exact Match
        result = self.l1.get(query)
        if result:
            self.stats["l1_hits"] += 1
            self.stats["avoided_llm_calls"] += 1
            return result, 1, True

        # L2: Semantic
        result = self.l2.get(query)
        if result:
            self.stats["l2_hits"] += 1
            self.stats["avoided_llm_calls"] += 1
            return result, 2, True

        # L3: LLM
        if llm_func:
            response = llm_func(query)
        else:
            response = f"[LLM 响应] 关于「{query[:30]}...」的回答"
        self.stats["llm_calls"] += 1

        # 写入缓存
        self.l1.set(query, response)
        self.l2.set(query, response)

        return response, 3, False

    def report(self) -> dict:
        total = sum([self.stats["l1_hits"], self.stats["l2_hits"],
                     self.stats["llm_calls"], self.stats["misses"]])
        total = max(total, 1)
        return {
            "total_queries": total,
            "l1_hit_rate": f"{self.stats['l1_hits'] / total:.1%}",
            "l2_hit_rate": f"{self.stats['l2_hits'] / total:.1%}",
            "llm_fallback_rate": f"{self.stats['llm_calls'] / total:.1%}",
            "total_cache_hit": (f"{(self.stats['l1_hits'] + self.stats['l2_hits']) / total:.1%}"),
            "avoided_llm_calls": self.stats["avoided_llm_calls"],
            "l1_size": len(self.l1._store),
            "l2_size": len(self.l2.queries),
        }


"""
28.3 Token 预算管理系统
━━━━━━━━━━━━━━━━━━━━━━━

类比：手机流量套餐

  每月 50GB：
    - 日常浏览（微信/网页）→ 低消耗
    - 看视频 → 高消耗
    - 快超标 → 限速 → 降级策略

  Agent Token 预算：
    每月 100M tokens：
      - 低风险任务 → 经评测的 economy / balanced 候选
      - 高风险任务 → high-capability 或人工处理
      - 快超标 → 限流、排队或降级，但不能绕过质量/安全下限

Token 预算的 4 个层级：
  1. 用户级：每人每月 N tokens
  2. 会话级：每会话 N tokens
  3. 请求级：每次 N tokens
  4. 步骤级：每步 N tokens（防死循环）
"""


class TokenBudget:
    """Token 预算管理器 —— 类比流量套餐管理。"""

    def __init__(self, daily_limit: int = 1_000_000):
        self.daily_limit = daily_limit
        self.used_today = 0
        self.warning_threshold = 0.8
        self.critical_threshold = 0.95
        self.history = []

    def check(self, required_tokens: int) -> dict:
        """检查预算是否充足。

        Returns:
            预算状态。
        """
        projected = self.used_today + required_tokens
        pct = projected / self.daily_limit

        if pct >= 1.0:
            return {"allowed": False, "status": "exceeded",
                    "message": f"今日预算已用完 ({self.daily_limit:,} tokens)"}

        if pct >= self.critical_threshold:
            return {"allowed": True, "status": "critical",
                    "message": f"接近预算上限 ({pct:.0%})，建议用小模型"}

        if pct >= self.warning_threshold:
            return {"allowed": True, "status": "warning",
                    "message": f"预算已达 {pct:.0%}"}

        return {"allowed": True, "status": "ok"}

    def consume(self, tokens: int):
        """消耗预算。"""
        self.used_today += tokens
        self.history.append({
            "tokens": tokens,
            "remaining": self.daily_limit - self.used_today,
            "pct": self.used_today / self.daily_limit,
        })

    def daily_report(self) -> dict:
        """每日预算报告。"""
        pct = self.used_today / self.daily_limit
        return {
            "daily_limit": f"{self.daily_limit:,}",
            "used": f"{self.used_today:,}",
            "remaining": f"{self.daily_limit - self.used_today:,}",
            "usage_pct": f"{pct:.1%}",
            "status": "🟢" if pct < 0.8 else ("🟡" if pct < 0.95 else "🔴"),
        }


def demo_cache():
    """演示三级缓存。"""
    print("=" * 60)
    print("  三级缓存演示")
    print("=" * 60)

    cache = ThreeLevelCache()

    # 模拟 FAQ 场景（高重复率）
    queries = [
        "退货流程是什么？",
        "如何修改密码？",
        "退货流程是什么？",        # ← 重复（L1命中）
        "退货的步骤是啥？",         # ← 语义相似（L2命中）
        "退货流程是什么？",        # ← 重复（L1命中）
        "怎么联系客服？",          # ← 新查询
        "如何修改密码？",          # ← 重复（L1命中）
        "怎么退换货物？",          # ← 语义相似（L2命中）
        "订单在哪里查询？",        # ← 新查询
        "退货流程是什么？",        # ← 重复（L1命中）
    ]

    for q in queries:
        response, level, cached = cache.query(q)
        icon = "🎯" if cached else "🆕"
        print(f"  {icon} L{level} 「{q[:15]}...」→ {response[:40]}...")

    print(f"\n  📊 缓存报告: {json.dumps(cache.report(), indent=2)}")


def demo_token_budget():
    """演示 Token 预算管理。"""
    print("\n" + "=" * 60)
    print("  Token 预算管理演示")
    print("=" * 60)

    budget = TokenBudget(daily_limit=10000)

    requests = [500, 2000, 4000, 3000, 1000]
    for i, tokens in enumerate(requests):
        check = budget.check(tokens)
        status = check["status"]
        icon = "✅" if check["allowed"] else "❌"
        status_msgs = {
            "ok": "预算正常",
            "warning": f'预算已达 {(budget.used_today + tokens) / budget.daily_limit:.0%}',
            "critical": f'接近预算上限 ({(budget.used_today + tokens) / budget.daily_limit:.0%})',
            "exceeded": f'今日预算已用完 ({budget.daily_limit:,} tokens)',
        }
        msg = status_msgs.get(status, "未知状态")
        print(f"  请求{i+1} {icon} {tokens:,}tokens → {status}: {msg}")
        if check["allowed"]:
            budget.consume(tokens)

    report = budget.daily_report()
    print(f"\n  📊 日预算报告:")
    for k, v in report.items():
        print(f"    {k}: {v}")


"""
28.4 缓存的进阶策略
━━━━━━━━━━━━━━━━━━━

1. 分层 TTL
   TTL 由数据源新鲜度和风险决定；即使是 Exact Match 也不能默认永久有效

2. 缓存预热
   上线前用常见问题预填充缓存

3. 缓存失效
   key 包含 tenant、权限、模型、prompt、tool/schema 与数据版本
   知识库更新 → 按依赖关系精准失效，无法追踪依赖时宁可不命中

4. 禁止或谨慎缓存
   个性化/含敏感信息、实时数据、权限相关回答和有副作用的工具结果默认不复用
   语义缓存命中必须经过意图、实体、时间范围和权限一致性检查

5. 监控指标
   命中率 / 误命中率 / 数据陈旧率 / 避免调用数 / LLM fallback 比例


28.5 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. 两级缓存 + LLM fallback：Exact → Semantic → LLM
2. Token 预算 = 流量套餐管理
3. 缓存收益取决于可缓存性、命中率、误命中成本和失效策略

面试速记：
  「Agent 成本怎么优化？」
  → Exact/Semantic 缓存命中率与误命中率从标注日志测量
  → 模型路由（简单用小模型，复杂用大模型）
  → Token 预算管理（防超支）
  → 同时报告质量、风险、延迟与实际账单
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第28章：语义缓存与 Token 优化                          ║")
    print("║  Exact Match · Semantic Cache · Token Budget        ║")
    print("╚══════════════════════════════════════════════════════╝")
    demo_cache()
    demo_token_budget()
    print("\n▶ 成本优化组合拳")
    for item in [
        "Exact/Semantic 缓存 → 只缓存允许复用且通过正确性验证的结果",
        "模型路由 → 按评测门槛选择候选，不假设固定流量比例",
        "Token 预算 → 防止单用户消耗超标",
        "组合收益 → 用真实账单减去缓存、失效和误命中处理成本",
    ]:
        print(f"  💰 {item}")
    print("\n✅ 第28章完成！🎓 全部课程构建完成！")
