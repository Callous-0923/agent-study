"""
第33章：Prompt Caching & 推理优化 —— 省钱的工程手段
=====================================================

📌 本章目标：
  1. 理解 Anthropic Prompt Caching 的原理和使用方式
  2. 掌握类比：Prompt Caching vs CDN vs 语义缓存
  3. 学会 KV-Cache 共享和推测解码的概念
  4. 量化各种缓存策略的收益

📌 面试高频点：
  - 「Anthropic 的 Prompt Caching 是怎么工作的？」
  - 「Prompt Caching 和语义缓存有什么区别？」
  - 「推测解码是什么？对 Agent 有什么价值？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
综合 Anthropic Prompt Caching API + 推理优化论文
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


33.1 缓存的三层体系
━━━━━━━━━━━━━━━━━━

┌──────────────────┬──────────────┬──────────┬──────────┐
│ 层                │     原理      │  命中率   │ 节省幅度  │
├──────────────────┼──────────────┼──────────┼──────────┤
│ L1: 语义缓存      │ 相似查询匹配   │ 30-50%   │ 100%     │
│ L2: Prompt Caching│ 相同前缀重用   │ 60-90%   │ 90%      │
│ L3: KV-Cache 共享 │ 跨请求共享状态  │ 取决于场景 │ 50-80%   │
└──────────────────┴──────────────┴──────────┴──────────┘

33.2 Anthropic Prompt Caching
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本质：对 System Prompt 和长上下文做缓存。

工作原理：
  1. 标记需要缓存的 content block（加上 cache_control）
  2. 相同前缀的请求，Claude 跳过重复的编码计算
  3. 输入 Token 费用降低 90%

使用示例：
  response = client.beta.prompt_caching.messages.create(
      model="claude-sonnet-4-20250514",
      system=[{
          "type": "text",
          "text": system_prompt,
          "cache_control": {"type": "ephemeral"},  ← 标记缓存
      }],
      messages=[...],
  )

缓存 min_token 要求：
  - 最少 1024 tokens 才能被缓存
  - 缓存 TTL: 5 分钟（无活跃使用则失效）


33.3 什么时候用 Prompt Caching？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

最适用的场景：
  ✓ Agent 的多轮对话（System Prompt 不变）
  ✓ 大量 Tool Definitions（每次调用都带同样的工具列表）
  ✓ 长文档 QA（同一文档多次提问）

不太适合的场景：
  ✗ 每次请求都是完全不同的上下文
  ✗ 短对话（System Prompt < 1024 tokens）
"""

import time
import hashlib


class PromptCacheSimulator:
    """模拟 Prompt Caching 的工作原理。"""

    def __init__(self):
        self.cache = {}  # {prefix_hash: cached_tokens}
        self.hits = 0
        self.misses = 0

    def call_llm(self, system_prompt: str,
                 user_message: str) -> dict:
        """模拟带缓存的 LLM 调用。

        Returns:
            含成本信息的字典。
        """
        prefix_hash = hashlib.md5(
            system_prompt.encode()
        ).hexdigest()

        prefix_tokens = len(system_prompt) // 4
        user_tokens = len(user_message) // 4
        total_tokens = prefix_tokens + user_tokens

        if prefix_hash in self.cache:
            self.hits += 1
            cost = user_tokens * 0.00001  # 仅计费新 token
            return {
                "cached": True,
                "tokens_charged": user_tokens,
                "total_tokens": total_tokens,
                "est_cost": round(cost, 6),
            }

        self.misses += 1
        self.cache[prefix_hash] = True
        cost = total_tokens * 0.00001
        return {
            "cached": False,
            "tokens_charged": total_tokens,
            "total_tokens": total_tokens,
            "est_cost": round(cost, 6),
        }

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hit_rate": f"{self.hits / max(total, 1):.0%}",
            "total_calls": total,
            "cache_size": len(self.cache),
        }


"""
33.4 推测解码 (Speculative Decoding)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原理：用小模型「猜测」大模型的输出，大模型验证。

流程：
  1. 小模型快速生成 N 个候选 token
  2. 大模型一次性验证这 N 个 token
  3. 通过率 > 80% → 速度快 2-3x

对 Agent 的价值：
  简单步骤（如「继续执行下一步」）→ 小模型足以预测
  复杂决策（如「该用什么工具」）→ 仍需要大模型


33.5 成本优化的组合策略
━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────┐
  │                                             │
  │  Ch26 模型路由  → 80% 请求用小模型           │
  │  Ch28 语义缓存  → 30-50% 免调 LLM            │
  │  Ch33 Prompt Caching → 90% 节省前缀 Token    │
  │                                             │
  │  三者叠加 → 综合成本降低 60-85%               │
  │                                             │
  └─────────────────────────────────────────────┘


33.6 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. Prompt Caching: 缓存 System Prompt，输入 Token 费降 90%
2. KV-Cache 共享: 跨请求复用 LLM 内部状态
3. 推测解码: 小模型猜 + 大模型验证，快 2-3x

面试速记：
  「Prompt Caching 怎么工作？」
  → 标记相同前缀 → Anthropic 跳过编码 → 输入 Token 费降 90%
  → 最少 1024 tokens 才能缓存 → TTL 5 分钟
"""


def demo_cache_simulator():
    print("=" * 60)
    print("  Prompt Caching 模拟")
    print("=" * 60)

    sim = PromptCacheSimulator()

    system = "你是一个专业的 AI 助手，需要详细回答用户的问题。请保持礼貌和专业。" * 10
    queries = [
        "今天天气怎么样？",
        "帮我写一段 Python 代码",
        "推荐几本 AI 书籍",
        "今天天气怎么样？",  # 重复查询，但 System Prompt 不变
    ]

    for q in queries:
        result = sim.call_llm(system, q)
        icon = "🎯 缓存命中" if result["cached"] else "🆕 首次调用"
        print(f"  {icon} | 计费: {result['tokens_charged']} tokens "
              f"| 成本: ${result['est_cost']:.6f}")

    s = sim.stats()
    print(f"\n  📊 缓存统计: 命中率 {s['hit_rate']}, "
          f"总调用 {s['total_calls']} 次")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第33章：Prompt Caching & 推理优化                     ║")
    print("║  Anthropic Cache · KV共享 · 推测解码                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    demo_cache_simulator()
    print("\n▶ 三层缓存体系")
    print("-" * 50)
    for name, benefit in [
        ("L1 语义缓存", "相似查询匹配，100% 节省"),
        ("L2 Prompt Caching", "相同前缀重用，90% 节省"),
        ("L3 KV-Cache 共享", "跨请求状态共享，50-80% 节省"),
    ]:
        print(f"  {name:18s} → {benefit}")
    print("\n✅ 第33章完成！")
