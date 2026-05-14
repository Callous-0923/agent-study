"""
第21章：Streaming & 实时 Agent 架构 —— 让用户不用等待的 Agent
===============================================================

📌 本章目标：
  1. 理解 Event-Driven Architecture 在 Agent 系统中的应用
  2. 掌握 Agent 执行中「动态中断」的架构设计
  3. 深入理解 Streaming Tool Calling 的状态机管理
  4. 学会用 EventBus 模式实现解耦的 Agent 架构
  5. 理解背压控制 (Backpressure) 和流控策略

📌 面试高频点：
  - 「Agent 怎么实现用户中途打断并修改方向？」
  - 「Streaming 模式下怎么处理 Tool Calling？」
  - 「为什么 Agent 需要事件驱动架构？」
  - 「EventBus 在 Agent 架构中扮演什么角色？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
综合来源：
  - Claude Code h2A 机制 (Ch8)
  - Event-Driven Agent Architecture (Sandipan Haldar, 2025)
  - A2A Streaming Protocol Design
  - 腾讯云 Agent 实时推理设计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


21.1 为什么 Agent 需要实时架构？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统「请求-响应」模式的问题：

  用户 → 发送消息 → [等待 30 秒...] → 收到完整回复

  Agent 执行过程中：
    - 用户不知道 Agent 在做什么
    - 用户发现方向不对，但只能等 Agent 完成再纠正
    - 每个步骤都是「黑盒」

实时架构的目标：

  用户 → 发送消息 → 实时看到 Agent 的思考过程 →
         中途可以修正方向 → 最终得到结果

核心能力：
  1. 流式输出：LLM 每生成一个 token 就推送给用户
  2. 中途介入：用户可以在 Agent 执行中途发送消息
  3. 工具调用可见：用户能看到 Agent 在调什么工具
  4. 可中断：用户可以随时停止 Agent 的任务

这是 Claude Code h2A 机制（Ch8）和 Ch13 SSE/WebSocket 的
「为什么」解释 —— 本章讲架构设计层面。


21.2 事件驱动架构 (EDA) —— Agent 系统的骨架
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心思想：
  传统架构：线性调用 → 阻塞等待 → 同步返回
  事件驱动：发布事件 → 异步处理 → 流式推送

Agent 的关键事件类型：

  ┌─────────────────────────────────────────────┐
  │              Agent 事件体系                   │
  ├──────────────┬──────────────────────────────┤
  │ 事件类型       │         说明                 │
  ├──────────────┼──────────────────────────────┤
  │ user_input   │ 用户发送消息                   │
  │ thinking     │ Agent 思考中                   │
  │ token_stream │ LLM 流式输出的一个 token       │
  │ tool_call    │ Agent 决定调用工具              │
  │ tool_result  │ 工具执行完成                    │
  │ interrupt    │ 用户中途介入/停止              │
  │ error        │ 发生错误                        │
  │ done         │ Agent 任务完成                  │
  └──────────────┴──────────────────────────────┘

关键组件：
  EventBus：事件总线，所有组件通过它通信
    → 解耦 LLM 调用、工具执行、UI 更新

  StateManager：状态管理器，唯一数据源
    → 记录 Agent 当前步骤、对话历史、工具执行状态

  StreamBridge：流式桥接
    → 将 LLM 的 streaming 输出桥接到前端


21.3 EventBus 模式 —— Agent 架构的核心
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：Agent 的多个组件（LLM 调用 / 工具执行 / 前端 UI）
      怎么高效通信而不互相耦合？

方案：EventBus —— 发布-订阅模式

  ┌──────────────────────────────────────────────────┐
  │                   EventBus                        │
  │                   (消息中枢)                       │
  │                                                    │
  │   pub/sub         pub/sub          pub/sub        │
  │   token_stream    tool_call         interrupt      │
  │      ↕               ↕                 ↕           │
  │  ┌────────┐    ┌──────────┐    ┌──────────┐     │
  │  │  LLM   │    │  Tool    │    │  User    │     │
  │  │ Service│    │ Executor │    │  Interface│     │
  │  └────────┘    └──────────┘    └──────────┘     │
  └──────────────────────────────────────────────────┘

EventBus 的优势：
  1. 解耦：LLM 不需要知道 UI 的存在
  2. 可测试：可以 Mock EventBus 单独测每个组件
  3. 可扩展：新组件只需订阅感兴趣的事件
  4. 可观测：所有事件都可以记录（天然 Tracing）

类比：
  就像办公室里的广播系统 ——
  没人需要知道「谁在听」，只需要「把消息广播出去」。
"""

import asyncio
import time
from typing import Any, Callable, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class AgentEvent:
    """Agent 事件基类。"""
    event_type: str
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    session_id: str = "default"


class EventBus:
    """Agent 事件总线 —— 发布-订阅模式的核心。

    所有 Agent 组件通过 EventBus 通信，不需要直接引用对方。
    这是 Event-Driven Agent 架构的基础设施。

    特性：
      1. 支持多个订阅者监听同一事件类型
      2. 支持通配符订阅 (*)
      3. 记录事件历史（用于调试和 Tracing）
      4. 异步处理（不阻塞发布者）
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._history: list[AgentEvent] = []
        self._max_history = 1000

    def subscribe(self, event_type: str,
                  callback: Callable[[AgentEvent], None]):
        """订阅事件类型。

        Args:
            event_type: 事件类型（支持 '*' 通配符）。
            callback: 事件处理回调函数。
        """
        self._subscribers[event_type].append(callback)

    def publish(self, event: AgentEvent):
        """发布事件（同步）。

        所有匹配的订阅者都会被通知。
        """
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # 精确匹配
        for callback in self._subscribers.get(event.event_type, []):
            callback(event)
        # 通配符匹配
        for callback in self._subscribers.get("*", []):
            callback(event)

    async def publish_async(self, event: AgentEvent):
        """异步发布事件（用于需要 await 的回调）。"""
        self._history.append(event)
        tasks = []
        for callback in self._subscribers.get(event.event_type, []):
            if asyncio.iscoroutinefunction(callback):
                tasks.append(asyncio.create_task(callback(event)))
            else:
                callback(event)
        for callback in self._subscribers.get("*", []):
            if asyncio.iscoroutinefunction(callback):
                tasks.append(asyncio.create_task(callback(event)))
            else:
                callback(event)
        if tasks:
            await asyncio.gather(*tasks)

    def get_history(self, event_type: str = None,
                    limit: int = 50) -> list[AgentEvent]:
        """获取事件历史（用于调试）。"""
        if event_type:
            filtered = [e for e in self._history
                        if e.event_type == event_type]
            return filtered[-limit:]
        return self._history[-limit:]


class AgentStateManager:
    """Agent 状态管理器 —— 单一数据源。

    使用纯函数（Reducer）更新状态，确保状态变更可预测。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = {
            "status": "idle",            # idle/thinking/acting/done/interrupted
            "current_step": 0,
            "tool_calls": [],
            "last_tool_result": None,
            "messages": [],
            "error": None,
            "interrupt_requested": False,
        }

    def reduce(self, event: AgentEvent):
        """纯函数式状态更新（Reducer 模式）。

        根据事件类型更新状态，没有副作用。
        """
        etype = event.event_type

        if etype == "thinking":
            self.state["status"] = "thinking"
            self.state["current_step"] += 1

        elif etype == "tool_call_start":
            self.state["status"] = "acting"
            self.state["tool_calls"].append({
                "tool": event.data.get("tool_name"),
                "args": event.data.get("args"),
                "status": "pending",
                "started_at": event.timestamp,
            })

        elif etype == "tool_call_end":
            if self.state["tool_calls"]:
                self.state["tool_calls"][-1]["status"] = "done"
                self.state["tool_calls"][-1]["result"] = event.data
            self.state["last_tool_result"] = event.data

        elif etype == "user_interrupt":
            self.state["interrupt_requested"] = True
            self.state["status"] = "interrupted"

        elif etype == "resume":
            self.state["interrupt_requested"] = False
            self.state["status"] = "thinking"

        elif etype == "done":
            self.state["status"] = "done"

        elif etype == "error":
            self.state["status"] = "error"
            self.state["error"] = event.data

    def should_interrupt(self) -> bool:
        """检查是否需要中断。"""
        return self.state["interrupt_requested"]

    def get_status_display(self) -> dict:
        """获取当前状态摘要（用于 UI 展示）。"""
        return {
            "status": self.state["status"],
            "step": self.state["current_step"],
            "active_tools": [t["tool"] for t in self.state["tool_calls"]
                             if t["status"] == "pending"],
        }


"""
21.4 动态中断 (Dynamic Interrupt) —— Agent 交互的质变
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是 Agent 从「自动化脚本」升级到「交互式伙伴」的关键能力。

传统模式：提交任务 → 等完成 → 看结果 → （不满意）→ 重新提交
中断模式：提交任务 → 观察进行中 → 途中发现方向不对 → 立即纠正

架构实现：
  1. EventBus 监听 interrupt 事件
  2. Agent 循环每步检查 should_interrupt()
  3. 如果中断 → 暂停执行 → 等待用户新指令
  4. 用户发送新指令 → resume 事件 → 继续执行

和 Claude Code h2A 的关系（Ch8）：
  h2A 是「双缓冲异步队列」
  EventBus 模式是 h2A 的「通用化实现」

关键在于中断的「粒度」：
  - 太粗：等当前工具调完才知道 → 延迟大
  - 太细：每收到一个 token 就检查 → 开销大
  - 最优：每个 Agent 步骤结束时检查（Claude Code 的做法）
"""


class InterruptibleAgent:
    """可中断 Agent —— 演示动态中断的完整流程。

    结合 EventBus + StateManager + 模拟 LLM。
    """

    def __init__(self, session_id: str = "demo"):
        self.bus = EventBus()
        self.state = AgentStateManager(session_id)
        self.cancelled = False

        # 注册事件处理
        self.bus.subscribe("user_interrupt",
                           lambda e: self.state.reduce(e))
        self.bus.subscribe("resume",
                           lambda e: self.state.reduce(e))

    def request_interrupt(self, reason: str = ""):
        """用户请求中断。"""
        self.bus.publish(AgentEvent(
            "user_interrupt",
            {"reason": reason},
            session_id=self.state.session_id,
        ))
        print(f"  ⏸️ 中断请求: {reason}")

    def resume(self, new_instruction: str = ""):
        """用户恢复执行（可带新指令）。"""
        self.bus.publish(AgentEvent(
            "resume",
            {"new_instruction": new_instruction},
            session_id=self.state.session_id,
        ))
        print(f"  ▶️ 恢复执行" +
              (f"（新指令: {new_instruction}）" if new_instruction else ""))

    async def run(self, task: str, max_steps: int = 10):
        """模拟 Agent 执行（可被中断）。"""
        print(f"  🎯 任务: {task}")

        for step in range(1, max_steps + 1):
            # 第1步：检查中断
            if self.state.should_interrupt():
                print(f"    ⏸️ Agent 在第 {step} 步暂停，等待用户...")
                yield {"type": "interrupted", "step": step}
                return
            if self.cancelled:
                yield {"type": "cancelled", "step": step}
                return

            # 第2步：思考
            self.bus.publish(AgentEvent("thinking"))
            yield {"type": "thinking", "step": step,
                   "message": f"正在分析第 {step} 步..."}
            await asyncio.sleep(0.3)

            # 第3步：模拟工具调用
            tool_name = f"tool_{step}"
            self.bus.publish(AgentEvent(
                "tool_call_start",
                {"tool_name": tool_name, "args": {"step": step}},
            ))
            yield {"type": "tool_call", "step": step,
                   "tool": tool_name}
            await asyncio.sleep(0.3)

            # 第4步：工具结果
            self.bus.publish(AgentEvent(
                "tool_call_end",
                f"工具 {tool_name} 执行成功",
            ))
            yield {"type": "tool_result", "step": step,
                   "result": f"{tool_name} 完成"}
            await asyncio.sleep(0.2)

        # 完成
        self.bus.publish(AgentEvent("done"))
        yield {"type": "done", "step": max_steps + 1,
               "message": "所有步骤完成"}


"""
21.5 背压控制 (Backpressure) —— 生产环境的隐性需求
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

场景：
  LLM 的 token 生成速度快于前端的渲染速度
  → 前端缓冲区满了 → 如果不处理 → 丢数据或 OOM

解决方案：

  1. 缓冲区控制
     前端维护一个固定大小的缓冲队列
     满了就暂停 SSE 流

  2. 自适应降频
     如果前端处理变慢 → 减少推送频率
     从「每 token 推送」降级到「每 5 token 推送」

  3. 生产者-消费者模式
     生产者（LLM）→ 队列 → 消费者（前端）
     队列有容量上限，满了暂停生产者

这就是 Claude Code h2A 的「智能背压控制」
（Ch8 提到：吞吐量 > 10,000 消息/秒 + 智能背压）。
"""


class BackpressureController:
    """背压控制器 —— 防止生产者速度超过消费者。

    生产者-消费者队列，内置流控机制。
    """

    def __init__(self, max_size: int = 100):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.dropped = 0  # 丢弃的事件数
        self.total = 0

    async def produce(self, event: AgentEvent) -> bool:
        """生产者：发布事件。

        Returns:
            是否成功入队。
        """
        self.total += 1
        try:
            self.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            return False

    async def consume(self) -> AgentEvent:
        """消费者：取出事件。"""
        return await self.queue.get()

    def stats(self) -> dict:
        """背压统计。"""
        return {
            "queue_size": self.queue.qsize(),
            "total_produced": self.total,
            "dropped": self.dropped,
            "drop_rate": self.dropped / max(self.total, 1),
        }


def demo_event_driven_agent():
    """演示事件驱动 Agent 的完整流程。"""
    print("=" * 60)
    print("  事件驱动 + 动态中断 演示")
    print("=" * 60)

    bus = EventBus()
    state = AgentStateManager("demo_session")

    # 注册事件处理器
    event_log = []
    bus.subscribe("*", lambda e: event_log.append(
        f"[{e.event_type}] {e.data}"
    ))

    bus.subscribe("thinking",
                  lambda e: state.reduce(e))
    bus.subscribe("tool_call_start",
                  lambda e: state.reduce(e))
    bus.subscribe("tool_call_end",
                  lambda e: state.reduce(e))

    # 模拟 Agent 执行流程
    print("\n  🔄 模拟 Agent 执行...")
    steps = [
        ("thinking", "正在分析任务..."),
        ("tool_call_start", {"tool_name": "search", "args": {"q": "AI"}}),
        ("tool_call_end", "找到 5 条结果"),
        ("thinking", "正在整理答案..."),
        ("user_interrupt", "等等，换一个方向！"),
    ]

    for etype, data in steps:
        event = AgentEvent(etype, data, session_id="demo_session")
        bus.publish(event)
        print(f"  📡 事件: {etype} → 状态: {state.get_status_display()['status']}")

    print(f"\n  📋 事件历史 ({len(event_log)} 条):")
    for entry in event_log:
        print(f"    {entry}")

    # 演示中断
    print(f"\n  ⚡ 中断检测: {state.should_interrupt()}")


"""
21.6 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. Event-Driven = Agent 实时交互的基础
   - EventBus 解耦各组件
   - StateManager 做唯一数据源
   - 事件驱动使中断、流式输出、工具调用可见成为可能

2. 动态中断
   - 用户可以在 Agent 执行中途「注入」新指令
   - 核心机制：EventBus + should_interrupt() 检查
   - 粒度控制：每步结束时检查（最优平衡）

3. 背压控制
   - 生产者（LLM）vs 消费者（前端）速度不匹配
   - 解决方案：有界队列 + 自适应降频
   - 这是从 Demo 到 Production 的关键一步

4. 架构和 Ch8/Ch13 的关系
   - Ch8 Claude Code h2A：一种具体的实现
   - Ch13 SSE/WebSocket：传输层实现
   - Ch21 EventBus：架构层抽象（本章）

面试速记：
  "Agent 怎么让用户可以中途介入？"
  → Event-Driven 架构 + EventBus
  → Agent 每步检查 interrupt 信号
  → h2A 双缓冲队列是 Claude Code 的具体实现
  → 关键是中断粒度：每步结束时检查
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第21章：Streaming & 实时 Agent 架构                   ║")
    print("║  EventBus · 动态中断 · 背压控制 · 状态机             ║")
    print("╚══════════════════════════════════════════════════════╝")

    demo_event_driven_agent()

    print("\n▶ Agent 关键事件类型")
    print("-" * 50)
    events = [
        ("user_input", "用户发送消息"),
        ("thinking", "Agent 思考中"),
        ("token_stream", "LLM 流式 token"),
        ("tool_call", "Agent 调用工具"),
        ("tool_result", "工具返回结果"),
        ("interrupt", "用户中途介入/停止"),
        ("error", "错误发生"),
        ("done", "任务完成"),
    ]
    for etype, desc in events:
        print(f"  {etype:15s} → {desc}")

    print("\n▶ 实时架构 Checklist")
    print("-" * 50)
    items = [
        "EventBus 解耦组件通信",
        "StateManager 单一数据源（纯函数 Reducer）",
        "每步结束检查中断信号",
        "背压控制防止生产者超过消费者",
        "事件历史用于 Tracing 和调试",
    ]
    for item in items:
        print(f"  ✅ {item}")

    print("\n✅ 第21章完成！")
