"""
第15章：Google A2A 协议 —— Agent 之间的「世界语」
====================================================

📌 本章目标：
  1. 深入理解 A2A (Agent-to-Agent) 协议的设计哲学
  2. 掌握 AgentCard / Task / Artifact 三大核心概念
  3. 理解 A2A 和 MCP 的分工关系（面试高频！）
  4. 体验完整的 Agent 发现→协商→执行流程
  5. 了解 Multi-Agent 协作的真实架构模式

📌 面试高频点：
  - A2A 和 MCP 分别解决什么问题？怎么配合使用？
  - AgentCard 里包含什么？为什么需要它？
  - A2A 的 Task 生命周期是怎样的？
  - 什么时候该用 A2A？什么时候不需要？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Google 于 2025年4月9日发布 A2A 协议
联合 50+ 企业：Salesforce、SAP、MongoDB、LangChain...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


15.1 A2A 解决了什么核心问题？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MCP 解决：「Agent 怎么和工具/数据库/API 交互？」
A2A 解决：「Agent 怎么和其他 Agent 交互？」

类比：
  MCP = 手机和充电器之间的 USB-C 协议
  A2A  = 手机之间的通信协议（像 4G/5G）

对比 MCP vs A2A（面试必问！）：

  ┌──────────────┬─────────────────────┬─────────────────────┐
  │     维度      │        MCP           │        A2A           │
  ├──────────────┼─────────────────────┼─────────────────────┤
  │ 解决什么      │ Agent ↔ 工具/数据     │ Agent ↔ Agent       │
  │ 发布方        │ Anthropic (2024.11)  │ Google (2025.04)    │
  │ 架构          │ Client-Host-Server   │ Client-Server       │
  │ 核心概念      │ Tools/Resources/Prompts│ AgentCard/Task/Artifact│
  │ 通信协议      │ JSON-RPC 2.0         │ JSON-RPC 2.0 + HTTP │
  │ 传输方式      │ stdio / SSE          │ HTTP / SSE          │
  │ 典型场景      │ 查数据库/调API       │ 多Agent协作/委派任务 │
  └──────────────┴─────────────────────┴─────────────────────┘

配合使用的示例（面试可以这样描述）：
  1. 用户 Agent 收到任务「预订去北京的出差行程」
  2. 用户 Agent 通过 MCP 调用数据库 → 查用户偏好
  3. 用户 Agent 通过 A2A → 发现「机票预订Agent」
  4. 用户 Agent 通过 A2A → 将「订机票」子任务委派给机票Agent
  5. 机票 Agent 通过 MCP → 调用航空公司 API
  6. 机票 Agent 通过 A2A → 返回预订结果


15.2 A2A 三大核心概念
━━━━━━━━━━━━━━━━━━━━━

1. AgentCard —— Agent 的「数字护照」

  每个 A2A Agent 都暴露一个 JSON 文件（/.well-known/agent.json），
  声明自己的能力、接口、安全要求：

  {
    "name": "TaxAgent",
    "description": "税务计算和合规分析 Agent",
    "url": "https://tax.example.com",
    "capabilities": {
      "streaming": true,
      "pushNotifications": false
    },
    "skills": [
      {
        "id": "tax_calculation",
        "name": "税务计算",
        "description": "支持中美跨境税务计算",
        "inputModes": ["text", "file"],
        "outputModes": ["text", "file"]
      }
    ],
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "interfaces": [
      {"url": "https://tax.example.com/a2a", "transport": "JSONRPC"}
    ]
  }

2. Task —— A2A 的工作单元

  Task 是整个协议的核心抽象：

  生命周期：pending → working → input-required → completed/failed/cancelled

  Task 数据结构：
  {
    "id": "task_001",
    "sessionId": "session_abc",
    "status": {"state": "working", "message": "正在计算税务..."},
    "artifacts": [...],
    "history": [...]  // 操作历史
  }

3. Artifact —— 跨 Agent 的成果物

  Task 完成后的产出，携带 MIME 类型和元数据：

  {
    "artifactId": "report_001",
    "name": "2025Q1税务报告.pdf",
    "mimeType": "application/pdf",
    "parts": [{"type": "file", "file": {...}}]
  }


15.3 A2A 的完整交互流程
━━━━━━━━━━━━━━━━━━━━━━━

  Client Agent                     Server Agent
      │                                │
      │── GET /.well-known/agent.json ─→│  (1) 发现：获取 AgentCard
      │←── AgentCard ─────────────────│
      │                                │
      │── POST /a2a tasks/send ──────→│  (2) 提交任务
      │   {message: "请计算这笔税的...", │
      │    artifacts: [...]}           │
      │←── Task {status: "working"} ──│
      │                                │
      │── POST /a2a tasks/get ────────→│  (3) 查询进度（轮询）
      │←── Task {status: "working"} ──│
      │                                │
      │── POST /a2a tasks/get ────────→│
      │←── Task {status: "completed",  │  (4) 获取结果
      │    artifacts: [计算结果]}      │
      │                                │

  也可以使用 SSE 做流式推送（替代轮询）：
      │── POST /a2a (SSE) ───────────→│
      │←── event: status-update ──────│  (实时推送状态变化)
      │←── event: artifact-update ────│  (实时推送生成内容)
      │←── event: task-complete ──────│


15.4 模拟 A2A Agent 生态系统
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import time
import hashlib
from typing import Optional
from enum import Enum


class TaskState(str, Enum):
    """A2A 协议定义的任务状态。"""
    PENDING = "pending"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class A2AAgentCard:
    """A2A AgentCard —— Agent 的自我声明文件。"""

    def __init__(self, name: str, description: str,
                 url: str, skills: list[dict]):
        self.name = name
        self.description = description
        self.url = url
        self.skills = skills
        self.capabilities = {
            "streaming": True,
            "pushNotifications": False,
        }
        self.defaultInputModes = ["text"]
        self.defaultOutputModes = ["text"]
        self.version = "0.2.6"

    def to_dict(self) -> dict:
        """序列化为标准 AgentCard JSON。"""
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "capabilities": self.capabilities,
            "skills": self.skills,
            "defaultInputModes": self.defaultInputModes,
            "defaultOutputModes": self.defaultOutputModes,
        }


class A2ATask:
    """A2A Task —— 协议的工作单元。"""

    def __init__(self, task_id: str, session_id: str):
        self.id = task_id
        self.session_id = session_id
        self.state = TaskState.PENDING
        self.status_message = "任务已创建，等待处理"
        self.artifacts = []
        self.history = []
        self._add_history("created", f"Task {task_id} created")

    def start(self):
        """开始执行任务。"""
        self.state = TaskState.WORKING
        self.status_message = "任务执行中..."
        self._add_history("started", "Task execution started")

    def complete(self, artifacts: list[dict]):
        """完成任务。"""
        self.state = TaskState.COMPLETED
        self.status_message = "任务已完成"
        self.artifacts = artifacts
        self._add_history("completed", f"Produced {len(artifacts)} artifacts")

    def fail(self, reason: str):
        """任务失败。"""
        self.state = TaskState.FAILED
        self.status_message = reason
        self._add_history("failed", reason)

    def to_dict(self) -> dict:
        """序列化为标准 Task JSON。"""
        return {
            "id": self.id,
            "sessionId": self.session_id,
            "status": {
                "state": self.state.value,
                "message": self.status_message,
            },
            "artifacts": self.artifacts,
            "history": self.history,
        }

    def _add_history(self, event: str, detail: str):
        self.history.append({
            "timestamp": time.time(),
            "event": event,
            "detail": detail,
        })


class SimulatedA2AServer:
    """模拟 A2A Server —— 演示 Agent-to-Agent 通信。

    完整的 A2A Server 实现：
      1. 暴露 AgentCard（GET /.well-known/agent.json）
      2. 接收任务（POST /a2a tasks/send）
      3. 查询任务状态（POST /a2a tasks/get）
      4. 取消任务（POST /a2a tasks/cancel）
    """

    def __init__(self, agent_card: A2AAgentCard):
        self.agent_card = agent_card
        self.tasks = {}

    def get_agent_card(self) -> dict:
        """返回 AgentCard（Agent 发现阶段）。"""
        return self.agent_card.to_dict()

    def send_task(self, message: str,
                  session_id: str = None) -> A2ATask:
        """接收并开始执行一个任务。

        Args:
            message: 任务描述。
            session_id: 会话 ID（不传则自动生成）。

        Returns:
            创建的 Task 对象。
        """
        task_id = hashlib.md5(
            f"{message}-{time.time()}".encode()
        ).hexdigest()[:16]
        if session_id is None:
            session_id = hashlib.md5(
                f"session-{time.time()}".encode()
            ).hexdigest()[:16]

        task = A2ATask(task_id, session_id)
        self.tasks[task_id] = task

        # 模拟任务执行
        task.start()

        # 找到匹配的 skill 并「执行」
        result_artifact = {
            "artifactId": f"artifact_{task_id}",
            "name": f"结果_{task_id[:8]}",
            "mimeType": "application/json",
            "parts": [{
                "type": "text",
                "text": json.dumps({
                    "summary": f"已完成任务: {message[:50]}...",
                    "agent": self.agent_card.name,
                    "timestamp": time.time(),
                }, ensure_ascii=False),
            }],
        }
        task.complete([result_artifact])

        return task

    def get_task(self, task_id: str) -> Optional[A2ATask]:
        """查询任务状态。

        Args:
            task_id: 任务 ID。

        Returns:
            Task 对象，不存在返回 None。
        """
        return self.tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务。

        Args:
            task_id: 任务 ID。

        Returns:
            是否成功取消。
        """
        task = self.tasks.get(task_id)
        if task and task.state in (TaskState.PENDING, TaskState.WORKING):
            task.state = TaskState.CANCELLED
            task.status_message = "任务已被取消"
            return True
        return False


class A2AClient:
    """模拟 A2A Client —— 作为「用户 Agent」去调用其他 Agent。

    代表 Agent 完成：
      1. 发现远程 Agent（获取 AgentCard）
      2. 评估 skills（是否匹配任务需求）
      3. 提交任务 + 获取结果
    """

    def __init__(self, name: str):
        self.name = name
        self.known_agents = {}  # {url: AgentCard}

    def discover(self, server: SimulatedA2AServer) -> A2AAgentCard:
        """发现一个 A2A Agent —— 获取其 AgentCard。

        Args:
            server: A2A Server 实例。

        Returns:
            AgentCard 字典。
        """
        card = server.get_agent_card()
        self.known_agents[card["url"]] = card
        return card

    def find_agent_for_skill(self, skill_keyword: str) -> Optional[dict]:
        """根据需求关键词找到匹配的 Agent。

        Args:
            skill_keyword: 技能关键词。

        Returns:
            匹配的 AgentCard，未找到返回 None。
        """
        for _, card in self.known_agents.items():
            for skill in card.get("skills", []):
                if skill_keyword in skill.get("name", "").lower() or \
                   skill_keyword in skill.get("description", "").lower():
                    return card
        return None

    def delegate_task(self, server: SimulatedA2AServer,
                      message: str) -> dict:
        """将任务委派给远程 Agent。

        完整的 A2A 委派流程：
          Discover → Evaluate → Send → Wait → Receive

        Args:
            server: A2A Server 实例。
            message: 任务描述。

        Returns:
            包含任务结果的字典。
        """
        result = {}

        # Step 1: 发现
        card = self.discover(server)
        result["agent"] = card["name"]
        result["skills"] = [s["name"] for s in card["skills"]]

        # Step 2: 提交任务
        task = server.send_task(message)
        result["task_id"] = task.id
        result["task_state"] = task.state.value

        # Step 3: 获取结果
        final_task = server.get_task(task.id)
        result["final_state"] = final_task.state.value
        result["artifacts"] = final_task.artifacts

        return result


def demo_a2a_ecosystem():
    """演示一个完整的 A2A 多 Agent 协作场景。"""
    print("=" * 60)
    print("  A2A 多 Agent 协作演示")
    print("=" * 60)

    # 创建 Agent 生态
    tax_agent = SimulatedA2AServer(A2AAgentCard(
        name="税务计算Agent",
        description="支持中美跨境税务计算和合规分析",
        url="https://tax.example.com",
        skills=[
            {
                "id": "tax_calculation",
                "name": "税务计算",
                "description": "计算个人所得税、企业所得税",
                "inputModes": ["text"],
                "outputModes": ["text"],
            },
            {
                "id": "tax_compliance",
                "name": "合规分析",
                "description": "分析税务合规风险",
                "inputModes": ["text"],
                "outputModes": ["text", "file"],
            },
        ],
    ))

    flight_agent = SimulatedA2AServer(A2AAgentCard(
        name="机票预订Agent",
        description="搜索和预订全球航班",
        url="https://flight.example.com",
        skills=[
            {
                "id": "flight_search",
                "name": "航班搜索",
                "description": "搜索可用的航班",
                "inputModes": ["text"],
                "outputModes": ["text"],
            },
            {
                "id": "flight_booking",
                "name": "航班预订",
                "description": "预订机票",
                "inputModes": ["text"],
                "outputModes": ["text"],
            },
        ],
    ))

    # 用户 Agent（协调者）
    user_agent = A2AClient("个人助理Agent")

    # 场景：用户需要出差
    print("\n  🎯 场景：用户需要去北京出差，请安排行程")

    # 子任务 1：税务相关查询
    print("\n  ── 子任务 1：税务查询 ──")
    tax_card = user_agent.discover(tax_agent)
    print(f"  发现 Agent: {tax_card['name']}")
    for skill in tax_card["skills"]:
        print(f"    技能: {skill['name']} - {skill['description']}")

    tax_result = user_agent.delegate_task(
        tax_agent, "计算2025年出差费用的税务抵扣"
    )
    print(f"  任务状态: {tax_result['task_state']} → {tax_result['final_state']}")
    artifact_text = tax_result["artifacts"][0]["parts"][0]["text"]
    print(f"  结果: {json.loads(artifact_text)['summary']}")

    # 子任务 2：预订机票
    print("\n  ── 子任务 2：预订机票 ──")
    flight_card = user_agent.discover(flight_agent)
    print(f"  发现 Agent: {flight_card['name']}")
    for skill in flight_card["skills"]:
        print(f"    技能: {skill['name']} - {skill['description']}")

    flight_result = user_agent.delegate_task(
        flight_agent, "预订2026年5月15日从上海到北京的机票"
    )
    print(f"  任务状态: {flight_result['task_state']} → {flight_result['final_state']}")
    artifact_text = flight_result["artifacts"][0]["parts"][0]["text"]
    print(f"  结果: {json.loads(artifact_text)['summary']}")

    # 展示 Agent 发现注册表
    print("\n  ── 已知 Agent 注册表 ──")
    for url, card in user_agent.known_agents.items():
        print(f"  📡 {card['name']} @ {url}")
        for s in card["skills"]:
            print(f"      └─ {s['name']}")


"""
15.5 A2A vs MCP 配合：完整的 Agent 协议栈
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

将它们组合起来：

  ┌────────────────────────────────────────────────────┐
  │                  Agent Application                   │
  │                                                      │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
  │  │ MCP Client│  │ MCP Client│  │ A2A Client│         │
  │  └─────┬────┘  └─────┬────┘  └─────┬────┘         │
  │        │             │             │                │
  └────────┼─────────────┼─────────────┼────────────────┘
           │             │             │
    MCP协议│      MCP协议│      A2A协议│
           ▼             ▼             ▼
  ┌────────────┐ ┌────────────┐ ┌────────────┐
  │ MCP Server  │ │ MCP Server  │ │ A2A Server  │
  │  数据库     │ │  文件系统   │ │  税务Agent  │
  └────────────┘ └────────────┘ └────────────┘
                                       │
                                 A2A协议│
                                       ▼
                                ┌────────────┐
                                │ A2A Server  │
                                │  机票Agent  │
                                └──────┬─────┘
                                       │
                                 MCP协议│
                                       ▼
                                ┌────────────┐
                                │ MCP Server  │
                                │ 航空公司API │
                                └────────────┘

面试速记版描述：
  "MCP 是 Agent 和工具之间的协议，A2A 是 Agent 和 Agent 之间的协议。
   实际系统中两者配合使用：Agent 通过 MCP 调用数据和工具，
   通过 A2A 把复杂子任务委派给其他专业的 Agent。"


15.5.1 A2A 的工程挑战 —— 理论很好，生产中要注意什么？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▍ Task 生命周期的边界情况（面试高分点）

  面试官问：「Task 状态机有哪些 tricky 的地方？」

  A2A Task 有 6 个状态：pending → working → input-required →
  completed/failed/cancelled。看起来简单，但真实工程中有很多边界：

  1. 取消正在 working 的 Task —— 不是「通知一下」就完了
     Server 需要：停止当前执行 → 回滚已做的操作 → 返回 cancelled
     （如果不回滚，Agent 可能已经把数据写到数据库了）

  2. input-required 状态下的超时 —— Client 不回应怎么办？
     A2A 规范没有规定超时行为 → 需要 Server 自己实现 timeout
     → 超时后自动 fail/cancel，避免资源泄漏

  3. 重复提交幂等性 —— 网络抖动导致 Client 发了 2 次 tasks/send
     → Server 需要根据 idempotency_key 去重
     → A2A 规范未强制要求幂等性，但生产环境必须实现

  4. Task 的「僵尸状态」—— working 状态但 Server 已崩溃
     → Client 无法获知 Server 崩溃了还是还在跑
     → 需要 heartbeat 机制：Server 定期更新 Task 的 last_heartbeat
     → Client 发现超时无 heartbeat → 判定失败

▍ Artifact 的大型文件传输问题

  A2A 的 Artifact 可以包含文件，但协议没有规定文件传输方式。
  
  小文件（< 10MB）：放在 Artifact 的 data URL 中（base64）
  大文件（> 10MB）：不应该 base64（膨胀 33%），应该传 URL
  → Agent A 把文件存到共享存储 → 在 Artifact 中放下载 URL
  → 这就是 MCP + A2A 的实际配合方式

▍ 安全模型 —— 跨组织 Agent 怎么互信？

  当前 A2A 的安全模型主要在「AgentCard 层面」：
    - AgentCard 声明需要什么认证（API Key / OAuth）
    - 但权限粒度不够 → 无法声明「这个 Task 需要什么权限」
  
  生产中的补救：
    - 在 Task payload 中自定义 auth_context 字段
    - 使用 OAuth Token Exchange（RFC 8693）做委托授权
    - Server 端做 Task 级别的权限校验


15.6 本章总结
━━━━━━━━━━━━━

核心要点回顾：

1. A2A = Agent 之间的「世界语」
   - Google 2025年4月发布，50+ 企业参与
   - 解决 Agent 间协作的标准化问题

2. 三大核心概念（面试反复问！）
   - AgentCard: Agent 的自我声明（skills/capabilities/interfaces）
   - Task: 工作单元（pending→working→completed/failed）
   - Artifact: 跨 Agent 的成果物封装

3. A2A 工程边界（面试进阶分）
   - Task 取消需要回滚已执行的操作
   - input-required 状态必须有超时兜底
   - 生产环境必须实现幂等键去重
   - heartbeat 机制防僵尸任务

4. A2A vs MCP —— 分工明确
   - MCP: Agent ↔ 工具/数据（Anthropic 发布）
   - A2A: Agent ↔ Agent（Google 发布）
   - 互补关系，不是竞争

5. 典型交互流程
   - 发现 (GET AgentCard)
   - 委派 (POST tasks/send)
   - 查询 (POST tasks/get，或 SSE 推送)
   - 获取 Artifact

面试速记：
  "A2A 是怎么工作的？"
  → AgentCard 声明能力 → Client 按需发现 → Task 委派
  → SSE/轮询跟踪进度 → 获取 Artifact
  → MCP 管工具，A2A 管 Agent 间协作
  → 生产注意：取消回滚、幂等键、heartbeat 超时
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第15章：Google A2A 协议深度剖析                        ║")
    print("║  AgentCard · Task · Artifact · Multi-Agent协作       ║")
    print("╚══════════════════════════════════════════════════════╝")

    demo_a2a_ecosystem()

    print("\n▶ A2A vs MCP 互补关系")
    print("-" * 50)
    pairs = [
        ("MCP", "Agent ↔ 工具/数据", "Anthropic 2024.11"),
        ("A2A", "Agent ↔ Agent", "Google 2025.04"),
    ]
    for name, purpose, source in pairs:
        print(f"  {name}: {purpose:30s} ({source})")

    print("\n▶ A2A Task 生命周期")
    for state in TaskState:
        print(f"  {state.value}")

    print("\n✅ 第15章完成！")
