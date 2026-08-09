"""
第33章：Prompt Caching——精确前缀复用与成本治理
================================================

内容核对：2026-08-01
说明：标注为模拟的实现与数值用于讲解概念，不代表真实 SDK、协议或基准结果。

📌 本章目标：
  1. 区分 Prompt Caching、响应缓存与语义缓存
  2. 掌握 OpenAI 与 Anthropic 当前的前缀缓存接口
  3. 学会设计稳定前缀、缓存边界和观测指标
  4. 用真实业务流量评估收益，而不是套用固定“节省比例”

33.1 三类容易混淆的缓存
────────────────────────

1. Prompt Caching（提供商侧前缀缓存）
   - 复用完全相同的已编码前缀或内部 KV 状态。
   - 后缀仍由模型处理并生成新响应。
   - 是否命中、最低前缀长度、TTL 和价格均由模型与提供商决定。

2. 响应缓存（应用侧 Exact Cache）
   - 相同且安全复用的请求直接返回旧响应，不调用模型。
   - 必须把租户、权限、模型、Prompt、工具和数据版本纳入缓存键。

3. 语义缓存（应用侧 Semantic Cache）
   - 相似问题可能复用答案。
   - 必须验证意图、实体、时间范围和权限；实时查询与副作用请求通常不能缓存。

Prompt Caching 的命中单位是“精确前缀”，不是“语义相似”。因此最重要的工程原则是：
把稳定内容放在前面，把用户问题、时间戳、随机 ID 等动态内容放在后面。

33.2 OpenAI：自动缓存与 GPT-5.6 显式断点
──────────────────────────────────────────────

截至 2026-08-01，OpenAI 对符合条件的近期模型自动启用 Prompt Caching。
GPT-5.6 及后续模型还支持显式缓存断点：
  - 在受支持的输入内容块上设置 prompt_cache_breakpoint；
  - 同一稳定前缀配合稳定的 prompt_cache_key；
  - prompt_cache_options.mode="explicit" 可关闭隐式断点；
  - GPT-5.6 的显式前缀最低为 1024 tokens；
  - prompt_cache_options.ttl 当前仅支持 "30m"，表示至少保留 30 分钟；
  - 写入量见 cache_write_tokens，命中量见 cached_tokens。

GPT-5.6+ 的缓存写入输入价格可能高于普通输入，读取价格则更低。不要只看命中率，
应同时计算“写入、读取、未缓存输入、输出”四类用量的实际账单。

真实 Responses API 请求骨架（需要已安装 openai，并设置 OPENAI_API_KEY）：

    from openai import OpenAI
    client = OpenAI()
    response = client.responses.create(
        model="gpt-5.6",
        prompt_cache_key="tenant-a:policy-v3",
        prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        input=[
            {
                "role": "system",
                "content": [{
                    "type": "input_text",
                    "text": stable_policy_and_tools,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }],
            },
            {"role": "user", "content": question},
        ],
    )
    cached = response.usage.input_tokens_details.cached_tokens

模型名称、价格和支持项会继续变化；生产代码应把模型 ID 放入配置，并以官方模型页为准。

33.3 Anthropic：cache_control 与可选 TTL
───────────────────────────────────────────

Anthropic Messages API 可在 system、messages 或 tools 的内容块上设置 cache_control。
默认临时缓存 TTL 为 5 分钟，也可选择 1 小时；最低可缓存 token 数随模型变化。

    import anthropic
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=anthropic_model_from_config,
        max_tokens=512,
        system=[{
            "type": "text",
            "text": stable_policy_and_tools,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }],
        messages=[{"role": "user", "content": question}],
    )

通过 usage 中的 cache_creation_input_tokens 与 cache_read_input_tokens 观察写入和读取。
不要把某个模型的最低长度、折扣或 TTL 推广为所有 Anthropic 模型的永久规则。

33.4 生产设计清单
──────────────────

  ✓ 稳定前缀：策略、长文档和工具定义放前面，动态问题放最后。
  ✓ 版本化：Prompt、工具、知识库或权限改变时更新 prompt_cache_key。
  ✓ 租户隔离：缓存键不得让不同租户或权限域意外复用敏感前缀。
  ✓ 保持字节稳定：工具顺序、JSON 序列化、图片和文本内容都可能影响命中。
  ✓ 可观测：分别记录写入 tokens、读取 tokens、未缓存 tokens、延迟和总成本。
  ✓ 实验：用代表性流量对照，不预设固定命中率或节省比例。
  ✓ 降级：缓存未命中或不可用时，正确性不能改变。

安全提醒：提供商侧缓存不等于应用侧授权。不要为了提高命中率，把本应隔离的用户数据、
权限上下文或密钥拼进共享前缀；还要遵守提供商的数据保留和 Zero Data Retention 说明。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass


@dataclass
class CacheEntry:
    expires_at: float
    estimated_prefix_tokens: int


class PrefixCacheSimulator:
    """教学模拟：演示“精确前缀 + TTL”，不模拟任何提供商的计费或内部 KV 实现。"""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, CacheEntry] = {}
        self.reads = 0
        self.writes = 0

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """仅供演示的粗略估算；真实用量以 API usage 字段为准。"""
        return max(1, len(text.encode("utf-8")) // 4)

    @staticmethod
    def _key(namespace: str, prefix: str) -> str:
        payload = f"{namespace}\0{prefix}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def request(self, namespace: str, stable_prefix: str, dynamic_suffix: str) -> dict:
        """返回缓存读写分类；dynamic_suffix 不参与前缀缓存键。"""
        now = time.monotonic()
        key = self._key(namespace, stable_prefix)
        entry = self.cache.get(key)
        hit = entry is not None and entry.expires_at > now

        prefix_tokens = self._estimate_tokens(stable_prefix)
        suffix_tokens = self._estimate_tokens(dynamic_suffix)
        if hit:
            self.reads += 1
            cache_write_tokens = 0
            cached_tokens = entry.estimated_prefix_tokens
        else:
            self.writes += 1
            self.cache[key] = CacheEntry(now + self.ttl_seconds, prefix_tokens)
            cache_write_tokens = prefix_tokens
            cached_tokens = 0

        return {
            "cache_hit": hit,
            "cache_write_tokens": cache_write_tokens,
            "cached_tokens": cached_tokens,
            "uncached_suffix_tokens": suffix_tokens,
        }

    def stats(self) -> dict:
        total = self.reads + self.writes
        return {
            "requests": total,
            "cache_reads": self.reads,
            "cache_writes": self.writes,
            "observed_read_rate": self.reads / total if total else 0.0,
        }


def demo_prefix_cache() -> None:
    print("=" * 64)
    print("Prompt Caching 教学模拟：同一租户与稳定前缀可复用")
    print("=" * 64)
    simulator = PrefixCacheSimulator(ttl_seconds=300)
    stable_prefix = "客服政策 v3；工具：query_order、create_refund；退款需要人工确认。"
    questions = ["查询订单 A100", "查询订单 A101", "解释退款条件"]

    for question in questions:
        result = simulator.request("tenant-a:policy-v3", stable_prefix, question)
        state = "读取缓存" if result["cache_hit"] else "写入缓存"
        print(
            f"{state:8s} | cached={result['cached_tokens']:3d} "
            f"write={result['cache_write_tokens']:3d} "
            f"suffix={result['uncached_suffix_tokens']:3d}"
        )

    print("统计：", simulator.stats())
    print("注意：上面 token 数为粗略教学估算，不代表 API 账单。")


if __name__ == "__main__":
    demo_prefix_cache()
