"""
第24章：Agent 可观测性 —— 生产环境的「监控仪表盘」
=====================================================

📌 本章目标：
  1. 理解 Agent 可观测性的 4 大支柱
  2. 掌握 Tracing 的核心设计模式
  3. 了解 LangSmith 和 LangFuse 的使用方式
  4. 学会构建 Agent 的监控仪表盘

📌 面试高频点：
  - 「Agent 的可观测性包含哪些维度？」
  - 「LangSmith 和 LangFuse 有什么区别？」
  - 「怎么追踪 Multi-Agent 的调用链？」
  - 「Level 2 Tracing 比 Level 1 多了什么？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基于 2025 年 Agent Observability 五大主流平台对比
+ OpenTelemetry GenAI SIG 标准
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


24.1 为什么 Agent 可观测性是「必需品」而非「可选项」？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统 API 监控：
  请求 → 响应 → 记录 HTTP 状态码 + 延迟 → 完成

Agent 监控的特殊性：
  一次「请求」包含 N 次 LLM 调用 + M 次工具调用
  错误可能在第 15 步才出现（难以定位）
  Multi-Agent 的调用链复杂（Agent A → Agent B → Agent C → ...）
  非确定性输出（同一输入，两次结果可能不同）

Agent 可观测性的 4 大支柱：

  ┌──────────────┬──────────────────────────────────────┐
  │    支柱        │              需要回答的问题            │
  ├──────────────┼──────────────────────────────────────┤
  │ 1. Tracing   │ Agent 执行了哪些步骤？每个步骤耗时？    │
  │ 2. Metrics   │ 整体成功率？P50/P99 延迟？Token消耗？   │
  │ 3. Evaluation │ 输出质量如何？是否满足用户需求？        │
  │ 4. Alerts    │ 什么情况需要人工介入？                  │
  └──────────────┴──────────────────────────────────────┘


24.2 Tracing —— 看见 Agent 的「每一根神经」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tracing 的层次模型：

  Level 1: 单次 LLM 调用 Trace
    记录: model / input_tokens / output_tokens / latency / status

  Level 2: Chain/Agent 级别 Trace
    记录: 完整的 Agent 执行链
    - 每个 tool_call: tool_name + args + result + latency
    - 每个 LLM 调用: system_prompt_snapshot + user_message
    - 每个步骤间的关系（父子 span）

  Level 3: Multi-Agent Trace
    记录: Agent 之间的消息传递
    - 哪个 Agent 把任务委派给了哪个 Agent
    - 委派的 context 是什么
    - 返回的 artifact 是什么

Trace 的核心数据结构：Span

  ┌──────────────────────────────────────────────┐
  │                  Trace (根)                    │
  │  ├── Span: Agent 步骤 1 (thinking)            │
  │  │   ├── Span: LLM 调用                       │
  │  │   │   ├── model: gpt-4o-mini              │
  │  │   │   ├── input_tokens: 1250              │
  │  │   │   ├── output_tokens: 86               │
  │  │   │   └── latency_ms: 1520                │
  │  │   └── Span: 工具调用                        │
  │  │       ├── tool: search                    │
  │  │       ├── args: {"query": "AI Agent"}      │
  │  │       ├── result_summary: "找到5条结果"     │
  │  │       └── latency_ms: 350                 │
  │  ├── Span: Agent 步骤 2 (tool_call)           │
  │  └── Span: Agent 步骤 3 (done)                │
  └──────────────────────────────────────────────┘
"""

import time
import json
import hashlib
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager


@dataclass
class TraceSpan:
    """Tracing Span —— 可观测性数据的基本单元。"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str                    # "llm_call" / "tool_call" / "agent_step"
    start_time: float
    end_time: Optional[float] = None
    attributes: dict = field(default_factory=dict)
    status: str = "ok"           # ok / error
    children: list = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000


class AgentTracer:
    """Agent 专用 Tracer —— 模拟 LangSmith/LangFuse 的功能。

    实现：
      1. Trace 管理（创建/结束/嵌套）
      2. LLM 调用自动记录
      3. 工具调用自动记录
      4. 跨 Agent 调用链
    """

    def __init__(self, project_name: str = "agent-study"):
        self.project_name = project_name
        self.traces: dict[str, list[TraceSpan]] = defaultdict(list)
        self._active_spans: dict[str, TraceSpan] = {}
        self.metrics = defaultdict(int)

    def start_trace(self, name: str,
                    session_id: str = None) -> str:
        """开始一个新的 Trace。

        Args:
            name: Trace 名称（如 "user_query"）。
            session_id: 会话 ID。

        Returns:
            trace_id。
        """
        trace_id = session_id or hashlib.md5(
            f"{name}-{time.time()}".encode()
        ).hexdigest()[:16]

        span = TraceSpan(
            trace_id=trace_id,
            span_id=f"span_{len(self.traces[trace_id])}",
            parent_span_id=None,
            name=name,
            start_time=time.time(),
        )
        self.traces[trace_id].append(span)
        self._active_spans[trace_id] = span
        return trace_id

    @contextmanager
    def span(self, trace_id: str, name: str, **attributes):
        """创建一个子 Span（上下文管理器）。

        用法:
          with tracer.span(tid, "llm_call", model="gpt-4o"):
              result = llm.invoke(...)

        Args:
            trace_id: 父 Trace ID。
            name: Span 名称。
            **attributes: 附加属性。
        """
        parent = self._active_spans.get(trace_id)

        span = TraceSpan(
            trace_id=trace_id,
            span_id=f"span_{len(self.traces[trace_id])}",
            parent_span_id=parent.span_id if parent else None,
            name=name,
            start_time=time.time(),
            attributes=attributes,
        )
        self.traces[trace_id].append(span)
        if parent:
            parent.children.append(span)

        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            span.end_time = time.time()
            span.attributes["latency_ms"] = span.latency_ms

    def log_llm_call(self, trace_id: str, model: str,
                     input_tokens: int, output_tokens: int,
                     latency_ms: float, success: bool = True):
        """记录一次 LLM 调用（便捷方法）。"""
        with self.span(trace_id, "llm_call",
                       model=model,
                       input_tokens=input_tokens,
                       output_tokens=output_tokens,
                       success=success) as span:
            span.end_time = span.start_time + latency_ms / 1000

        self.metrics["total_llm_calls"] += 1
        self.metrics["total_input_tokens"] += input_tokens
        self.metrics["total_output_tokens"] += output_tokens
        if not success:
            self.metrics["failed_llm_calls"] += 1

    def log_tool_call(self, trace_id: str, tool_name: str,
                      input_args: dict, output_summary: str,
                      latency_ms: float, success: bool = True):
        """记录一次工具调用（便捷方法）。"""
        with self.span(trace_id, "tool_call",
                       tool_name=tool_name,
                       input_args=str(input_args)[:200],
                       output_summary=output_summary[:200],
                       success=success) as span:
            span.end_time = span.start_time + latency_ms / 1000

        self.metrics["total_tool_calls"] += 1
        if not success:
            self.metrics["failed_tool_calls"] += 1

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """获取 Trace 的完整信息（用于调试和展示）。"""
        spans = self.traces.get(trace_id, [])
        if not spans:
            return None

        total_latency = sum(
            s.latency_ms for s in spans
            if s.end_time is not None
        )
        failed = sum(1 for s in spans if s.status == "error")

        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "total_latency_ms": round(total_latency, 1),
            "failed_spans": failed,
            "spans": [
                {
                    "name": s.name,
                    "latency_ms": round(s.latency_ms, 1),
                    "status": s.status,
                    "attributes": {
                        k: v for k, v in s.attributes.items()
                        if k in ("model", "tool_name", "input_tokens",
                                 "output_tokens", "success")
                    },
                    "children": len(s.children),
                }
                for s in spans
            ],
        }

    def get_metrics(self) -> dict:
        """获取聚合指标。"""
        return {
            "project": self.project_name,
            "traces": len(self.traces),
            "metrics": dict(self.metrics),
            "avg_tokens_per_call": (
                self.metrics["total_input_tokens"] /
                max(self.metrics["total_llm_calls"], 1)
            ),
            "tool_success_rate": (
                1 - self.metrics["failed_tool_calls"] /
                max(self.metrics["total_tool_calls"], 1)
            ) if self.metrics["total_tool_calls"] > 0 else 0,
        }

    def print_trace_tree(self, trace_id: str, indent: int = 0):
        """打印 Trace 树形结构（直观展示执行流程）。"""
        spans = self.traces.get(trace_id, [])
        if not spans:
            print("  无 Trace 数据")
            return

        # 找到根 Span
        root = None
        for s in spans:
            if s.parent_span_id is None:
                root = s
                break

        if root is None:
            root = spans[0]

        def _print_span(span: TraceSpan, level: int):
            prefix = "  " * level + ("├─ " if level > 0 else "")
            status_icon = "✅" if span.status == "ok" else "❌"
            attrs = ""
            if "model" in span.attributes:
                attrs += f" [{span.attributes['model']}]"
            if "tool_name" in span.attributes:
                attrs += f" [{span.attributes['tool_name']}]"
            print(f"{prefix}{status_icon} {span.name}{attrs} "
                  f"({span.latency_ms:.0f}ms)")

            for child in span.children:
                _print_span(child, level + 1)

        _print_span(root, 0)

    def print_dashboard(self):
        """打印监控仪表盘（模拟 Grafana 面板）。"""
        m = self.get_metrics()
        print("=" * 55)
        print("  📊 Agent 监控仪表盘")
        print("=" * 55)
        print(f"  项目: {m['project']}")
        print(f"  Traces 总数: {m['traces']}")
        print(f"  LLM 调用: {m['metrics'].get('total_llm_calls', 0)}")
        print(f"  总输入 Tokens: {m['metrics'].get('total_input_tokens', 0):,}")
        print(f"  总输出 Tokens: {m['metrics'].get('total_output_tokens', 0):,}")
        print(f"  平均 Tokens/次: {m['avg_tokens_per_call']:.0f}")
        print(f"  工具调用: {m['metrics'].get('total_tool_calls', 0)}")
        print(f"  工具成功率: {m['tool_success_rate']:.0%}")
        status = "🟢 正常" if m['metrics'].get('failed_llm_calls', 0) == 0 else "🔴 异常"
        print(f"  系统状态: {status}")
        print("-" * 55)


def demo_agent_tracing():
    """演示 Agent Tracing 的完整流程。"""
    print("=" * 60)
    print("  Agent Tracing 完整演示")
    print("=" * 60)

    tracer = AgentTracer(project_name="my-agent-app")

    # 模拟一个用户请求的完整 Agent 执行过程
    trace_id = tracer.start_trace("用户查询天气", session_id="session_abc")

    # LLM 调用 1：理解用户意图
    tracer.log_llm_call(trace_id, "gpt-4o-mini",
                        input_tokens=450, output_tokens=85,
                        latency_ms=1200)

    # 工具调用 1：搜索天气
    tracer.log_tool_call(trace_id, "search_weather",
                         {"city": "北京"}, "晴天 25°C",
                         latency_ms=320)

    # LLM 调用 2：整理回答
    tracer.log_llm_call(trace_id, "gpt-4o-mini",
                        input_tokens=650, output_tokens=120,
                        latency_ms=980)

    # 展示 Trace 树
    print("\n  🌲 Trace 树形结构:")
    tracer.print_trace_tree(trace_id)

    # 展示 Trace 详细数据
    trace_info = tracer.get_trace(trace_id)
    print(f"\n  📋 Trace 摘要: {trace_info['span_count']} 个 Span, "
          f"总延迟 {trace_info['total_latency_ms']}ms, "
          f"失败 {trace_info['failed_spans']}")

    # 模拟第二个请求（带工具失败）
    trace_id2 = tracer.start_trace("用户计算数学题", session_id="session_abc")
    tracer.log_llm_call(trace_id2, "gpt-4o-mini",
                        input_tokens=300, output_tokens=65,
                        latency_ms=800)
    tracer.log_tool_call(trace_id2, "calculator",
                         {"expr": "123+456"}, "",
                         latency_ms=50, success=False)
    tracer.log_tool_call(trace_id2, "calculator",
                         {"expr": "123+456"}, "579",
                         latency_ms=45, success=True)
    tracer.log_llm_call(trace_id2, "gpt-4o-mini",
                        input_tokens=400, output_tokens=70,
                        latency_ms=600)

    print(f"\n  🌲 第二个 Trace:")
    tracer.print_trace_tree(trace_id2)

    # Dashboard
    print(f"\n")
    tracer.print_dashboard()


"""
24.3 LangSmith vs LangFuse —— 两大平台对比
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────┬─────────────────────┬─────────────────────┐
  │     维度      │      LangSmith       │      LangFuse        │
  ├──────────────┼─────────────────────┼─────────────────────┤
  │ 开发者        │ LangChain 公司       │ 独立开源社区         │
  │ 开源          │ 部分开源             │ 完全开源 (MIT)       │
  │ 部署          │ SaaS (托管)          │ SaaS + 自托管 (Docker/K8s)│
  │ LangChain集成 │ 原生（环境变量开箱）   │ 通过回调集成         │
  │ 定价          │ 免费层 + Pro/Enterprise│ 免费层 + Cloud/Enterprise│
  │ 特色功能       │ Hub (Prompt共享)     │ Prompt Management    │
  │ 评测          │ LLM-as-Judge 评测     │ Dataset Runs 评测    │
  └──────────────┴─────────────────────┴─────────────────────┘

选型建议：
  - 重度用 LangChain/LangGraph → LangSmith（原生集成）
  - 需要自托管 + 完全开源 → LangFuse
  - 小团队快速上手 → LangSmith Cloud
  - 需要 Prompt 版本管理 → LangFuse


24.4 告警系统 —— 被忽视的关键
━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent 告警规则（面试时脱口而出！）：

  关键指标告警：
    ☐ 成功率 < 95% (1h 窗口) → PagerDuty/Slack 告警
    ☐ P99 延迟 > 60s → 延迟告警
    ☐ Token 消耗 > 预算 × 80% → 预算告警
    ☐ 工具调用失败率 > 10% → 工具故障告警
    ☐ 连续 N 次相同错误 → 死循环告警

  异常检测：
    ☐ 用户输入异常（Prompt Injection 检测触发）
    ☐ 输出异常（内容安全审核触发）
    ☐ 流量异常（Rate Limiting 触发）


24.5 Agent 可观测性 Checklist
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Tracing
  ☐ 每次 LLM 调用：model + tokens + latency
  ☐ 每次工具调用：tool_name + args + result + latency
  ☐ 父子 Span 关系（便于定位性能瓶颈）
  ☐ 错误 Span 标记 + 错误原因

✅ Metrics
  ☐ 成功率 Dashboard（按模型/按工具/按时段）
  ☐ P50/P95/P99 延迟趋势
  ☐ Token 消耗趋势（按天/按用户）
  ☐ 错误分布热力图

✅ Evaluation
  ☐ 离线评测集 + 自动回归
  ☐ LLM-as-Judge 质量评分
  ☐ 用户反馈收集

✅ Alerts
  ☐ 核心指标告警规则
  ☐ 异常检测
  ☐ 人工介入流程


24.6 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. Agent 可观测性 4 大支柱
   Tracing / Metrics / Evaluation / Alerts

2. Tracing 三层模型
   Level 1 (LLM调用) → Level 2 (Agent链) → Level 3 (Multi-Agent)

3. LangSmith vs LangFuse
   LangSmith: LangChain 原生集成
   LangFuse: 完全开源 + 自托管

面试速记：
  "Agent 怎么监控？"
  → 4 大支柱：Tracing + Metrics + Evaluation + Alerts
  → Tracing：记录每个 LLM 调用 + 工具调用
  → LangSmith/LangFuse 选型
  → 告警：成功率 < 95% → 通知
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第24章：Agent 可观测性                                 ║")
    print("║  Tracing · Metrics · Dashboard · LangSmith/LangFuse ║")
    print("╚══════════════════════════════════════════════════════╝")

    demo_agent_tracing()

    print("\n▶ LangSmith vs LangFuse")
    print("-" * 50)
    pairs = [
        ("开发者", "LangSmith: LangChain公司", "LangFuse: 独立开源社区"),
        ("开源", "LangSmith: 部分开源", "LangFuse: 完全开源 MIT"),
        ("部署", "LangSmith: SaaS (托管)", "LangFuse: SaaS + 自托管"),
        ("LangChain集成", "LangSmith: 原生(环境变量)", "LangFuse: 回调集成"),
        ("推荐场景", "LangSmith: 重度LangChain用户", "LangFuse: 自建/开源优先"),
    ]
    for dim, ls, lf in pairs:
        print(f"  {dim:12s} | {ls:30s} | {lf}")

    print("\n▶ 告警规则 Checklist")
    print("-" * 50)
    alerts = [
        "成功率 < 95% (1h) → PagerDuty/Slack",
        "P99 延迟 > 60s → 延迟告警",
        "Token 消耗 > 预算×80% → 预算告警",
        "工具调用失败率 > 10% → 工具故障",
        "连续 N 次相同错误 → 死循环告警",
    ]
    for a in alerts:
        print(f"   🔔 {a}")

    print("\n✅ 第24章完成！")
