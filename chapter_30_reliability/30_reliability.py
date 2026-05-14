"""
第30章：Agent 可靠性工程 —— 在生产环境中活下来
===============================================

📌 本章目标：
  1. 掌握 Agent 专用的可靠性模式（熔断/舱壁/重试/幂等/Saga）
  2. 理解 Agent 系统特有的故障模式
  3. 实现 Exponential Backoff 重试和熔断器
  4. 学会设计 Agent 的降级策略

📌 面试高频点：
  - 「Agent 在生产中出错了怎么办？」
  - 「什么是熔断器？怎么给 Agent 加熔断？」
  - 「幂等性在 Agent 工具调用中怎么保证？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
参考：Netflix Hystrix / Resilience4j / 分布式系统可靠性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


30.1 Agent 为什么需要专门的可靠性？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统 API 的故障模式：
  - 超时 → 重试
  - 500 → 降级

Agent 特有的故障模式：
  - LLM 返回幻觉参数 → 工具执行报错
  - Agent 陷入死循环 → 无限消耗 Token
  - 工具调用雪崩 → 一个失败导致后续全部失败
  - 上下文爆炸 → 对话历史过长 → LLM 行为异常

可靠性 = 系统在故障下仍能提供「可接受」的服务。


30.2 熔断器 (Circuit Breaker)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原理：当某个工具的失败次数超过阈值，直接跳过它
      而不是继续浪费时间调用。

三态模型：
  CLOSED → 正常调用
  OPEN   → 直接拒绝（熔断中），避免雪崩
  HALF_OPEN → 放行一个请求试探 → 成功则 CLOSED，失败则 OPEN

类比：保险丝
  电流过大 → 熔断 → 保护电路
  冷却后 → 恢复
"""

import time
import json
import random
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """熔断器 —— 防止对故障工具的重复调用。

    三态：
      CLOSED: 正常放行
      OPEN: 直接拒绝
      HALF_OPEN: 试探性放行一个请求
    """

    def __init__(self, name: str, failure_threshold: int = 3,
                 recovery_timeout: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
        self.total_calls = 0

    def call(self, func, *args, **kwargs) -> dict:
        """通过熔断器调用函数。

        Returns:
            含结果或熔断信息的字典。
        """
        self.total_calls += 1

        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                return {
                    "success": False,
                    "error": f"熔断器 {self.name} 已断开，请求被拒绝",
                    "state": self.state.value,
                }

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return {"success": True, "result": result}
        except Exception as e:
            self._on_failure()
            return {"success": False, "error": str(e)}

    def _on_success(self):
        self.failure_count = 0
        self.success_count += 1
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
        }


"""
30.3 指数退避重试 (Exponential Backoff)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

为什么不能用固定间隔重试？
  如果工具故障是因为过载 → 所有 Agent 同时重试 → 加重过载

指数退避：
  第1次重试：等待 1 秒
  第2次重试：等待 2 秒
  第3次重试：等待 4 秒
  ...
  + 随机抖动（jitter）防止多个 Agent 同步重试
"""


def retry_with_backoff(func, max_retries: int = 3,
                       base_delay: float = 1.0,
                       max_delay: float = 30.0) -> dict:
    """指数退避重试装饰器。

    Args:
        func: 要重试的函数。
        max_retries: 最大重试次数。
        base_delay: 基础延迟（秒）。
        max_delay: 最大延迟上限。

    Returns:
        执行结果字典。
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = func()
            return {"success": True, "result": result,
                    "attempts": attempt + 1}
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)
                time.sleep(delay + jitter)

    return {"success": False, "error": last_error,
            "attempts": max_retries + 1}


"""
30.4 幂等性保证
━━━━━━━━━━━━━━━━━

问题：Agent 重试工具调用时，可能重复执行已有副作用的操作。
      例如：send_email 重试 → 用户收到 2 封邮件。

方案：幂等键 (Idempotency Key)

  def send_email_idempotent(to, subject, body, idempotency_key):
      if already_sent(idempotency_key):
          return cached_result
      result = send_email(to, subject, body)
      cache(idempotency_key, result)
      return result

每个工具调用携带唯一键，服务端检查去重。
"""


class IdempotencyGuard:
    """幂等性保护器。"""

    def __init__(self):
        self._cache = {}

    def execute(self, key: str, func, *args, **kwargs) -> dict:
        """幂等执行：相同 key 只执行一次。

        Args:
            key: 幂等键。
            func: 要执行的函数。

        Returns:
            执行结果。
        """
        if key in self._cache:
            return {"cached": True, **self._cache[key]}

        try:
            result = func(*args, **kwargs)
            self._cache[key] = {"success": True, "result": result}
            return {"cached": False, "success": True, "result": result}
        except Exception as e:
            self._cache[key] = {"success": False, "error": str(e)}
            return {"cached": False, "success": False, "error": str(e)}


"""
30.5 降级策略 (Graceful Degradation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当核心能力不可用时，提供降级方案：

┌──────────────────┬──────────────────────────┐
│ 故障               │          降级             │
├──────────────────┼──────────────────────────┤
│ LLM 不可用         │ 返回预置的 FAQ 回答        │
│ 搜索工具熔断       │ 用本地缓存结果代替         │
│ 支付工具超时       │ 排队异步处理 + 通知用户      │
│ 上下文过大         │ 触发压缩 + 移除旧对话        │
└──────────────────┴──────────────────────────┘
"""


"""
30.6 可靠性 Checklist
━━━━━━━━━━━━━━━━━━━━

✅ 熔断器
  ☐ 每个外部工具都有熔断器保护
  ☐ 熔断阈值可配置（默认 3 次失败 / 30s 恢复）

✅ 重试
  ☐ 指数退避 + 随机抖动
  ☐ 最大重试次数限制（默认 3 次）
  ☐ 只重试「可重试」的错误（429/503，不重试 400/401）

✅ 幂等
  ☐ 写操作使用幂等键
  ☐ 读操作天然幂等（不需要额外保护）

✅ 降级
  ☐ LLM 不可用时的兜底回答
  ☐ 关键工具不可用时的替代方案

✅ 超时
  ☐ 每次 LLM 调用设置超时（默认 60s）
  ☐ 每次工具调用设置超时（默认 30s）

面试速记：
  「Agent 怎么保证可靠性？」
  → 熔断器防雪崩 + 退避重试 + 写入幂等 + 降级兜底 + 超时控制
"""


def demo_reliability():
    """演示可靠性机制。"""
    print("=" * 60)
    print("  Agent 可靠性工程演示")
    print("=" * 60)

    # 1. 熔断器
    print("\n  ── 熔断器 ──")
    cb = CircuitBreaker("search_tool", failure_threshold=2)

    def failing_search():
        raise ConnectionError("服务不可用")

    def working_search():
        return "搜索结果: ..."

    for i in range(5):
        result = cb.call(failing_search if i < 3 else working_search)
        status = "✅" if result["success"] else "❌"
        print(f"    调用{i+1} {status} 状态={cb.state.value}"
              f" 失败={cb.failure_count}")

    # 2. 重试
    print("\n  ── 指数退避重试 ──")
    call_count = [0]

    def flaky_api():
        call_count[0] += 1
        if call_count[0] < 3:
            raise RuntimeError("临时故障")
        return "成功!"

    result = retry_with_backoff(flaky_api, max_retries=3)
    print(f"    结果: {result['success']} | 尝试次数: {result['attempts']}")

    # 3. 幂等
    print("\n  ── 幂等性 ──")
    guard = IdempotencyGuard()
    key = "send_email_abc123"

    def send():
        print("    → 实际发送邮件...（仅执行一次）")
        return "邮件已发送"

    r1 = guard.execute(key, send)
    print(f"    第1次: cached={r1['cached']} success={r1['success']}")
    r2 = guard.execute(key, send)
    print(f"    第2次: cached={r2['cached']} (已缓存，未重复执行)")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第30章：Agent 可靠性工程                               ║")
    print("║  熔断器 · 退避重试 · 幂等 · 降级 · 超时                ║")
    print("╚══════════════════════════════════════════════════════╝")
    demo_reliability()
    print("\n▶ 可靠性 Checklist")
    print("-" * 50)
    for item in [
        "熔断器: 每个外部工具 → 3次失败/30s 恢复",
        "重试: 指数退避 + 随机抖动 + 最大3次",
        "幂等: 写操作携带 idempotency_key",
        "降级: LLM不可用 → 兜底回答",
        "超时: LLM 60s / Tool 30s",
    ]:
        print(f"  ✅ {item}")
    print("\n✅ 第30章完成！")
