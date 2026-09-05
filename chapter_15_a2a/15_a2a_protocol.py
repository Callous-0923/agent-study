"""
第15章：A2A 1.0 —— Agent 发现、消息与任务协作
==============================================

内容核对：2026-08-01
说明：协议字段以 A2A v1.0 为核对基线；本章 Python Server 是教学模拟，
不替代官方 SDK，也未实现真实 REST、gRPC、JSON-RPC 网络绑定或 JWS 验签。
官方 v1.0 变更：https://a2a-protocol.org/latest/whats-new-v1/

📌 本章目标：
  1. 理解 A2A 与 MCP、Function Calling 的边界
  2. 掌握 AgentCard、Message、Task、Artifact 四个核心对象
  3. 熟悉 SendMessage / GetTask / ListTasks / CancelTask 等 v1.0 操作
  4. 理解流式事件、长任务、身份授权和多租户隔离
  5. 识别从 v0.x 迁移到 v1.0 的破坏性变化


15.1 A2A 解决什么问题？
━━━━━━━━━━━━━━━━━━━━━━━

当一个 Agent 需要把任务委派给另一个独立服务时，仅有“调用函数”还不够：
远端 Agent 可能需要数分钟处理、请求补充信息、流式产生中间结果，并最终交付
多个 Artifact。A2A 为这类跨服务协作定义发现、消息、任务状态和产物模型。

  Function Calling：模型选择本进程可用函数
  MCP             ：AI Host 发现并调用工具、资源和提示模板
  A2A             ：独立 Agent 服务之间发现能力并交换 Message / Task

常见组合：主 Agent 通过 A2A 委派“生成财务报告”，财务 Agent 内部再通过 MCP
读取数据库和模板。协议可以组合，但不应把远端 Agent 伪装成无状态小函数。


15.2 AgentCard v1.0
━━━━━━━━━━━━━━━━━━

公开 AgentCard 通常通过 well-known URI 发现。v1.0 的重要变化是：
  - endpoint 不再放在顶层 `url`
  - `protocolVersion` 位于每个 AgentInterface
  - `supportedInterfaces[]` 合并旧 preferred/additional transport 字段
  - `capabilities.extendedAgentCard` 表示可获取鉴权后的扩展卡
  - 可使用 JCS + JWS 验证签名；不要仅信任下载地址

示例（字段按实际 SDK 生成，勿手写长期维护）：

  {
    "name": "Report Agent",
    "description": "生成审计报告",
    "version": "2.3.0",
    "supportedInterfaces": [{
      "url": "https://agents.example.com/report",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    }],
    "capabilities": {"streaming": true, "extendedAgentCard": true},
    "defaultInputModes": ["text/plain", "application/json"],
    "defaultOutputModes": ["text/markdown", "application/json"],
    "skills": [{
      "id": "audit-report",
      "name": "审计报告",
      "description": "根据已授权的数据生成报告",
      "tags": ["audit", "report"]
    }]
  }


15.3 v1.0 对象模型
━━━━━━━━━━━━━━━━━━━

Message：一次用户或 Agent 消息，role 使用 ROLE_USER / ROLE_AGENT。
Part   ：统一内容结构，通过 text / raw / url / data 成员区分，不再使用 kind。
Task   ：可跟踪的工作单元，含 id、contextId、status、history、artifacts。
Artifact：Agent 交付的结构化结果，由一个或多个统一 Part 组成。

Task 状态使用 SCREAMING_SNAKE_CASE：
  TASK_STATE_SUBMITTED / WORKING / COMPLETED / FAILED / CANCELED /
  REJECTED / INPUT_REQUIRED / AUTH_REQUIRED

时间戳使用 ISO 8601 UTC，精确到毫秒。服务端必须按已认证调用者限制 Task 可见性。


15.4 核心操作与流式事件
━━━━━━━━━━━━━━━━━━━━━━━

  SendMessage          ：发送消息，返回 Message 或 Task
  SendStreamingMessage ：发送消息并接收有序事件流
  GetTask              ：读取调用者可见的 Task
  ListTasks            ：分页列出调用者可见的 Task
  CancelTask           ：请求取消允许取消的 Task
  SubscribeToTask      ：重新订阅仍在运行的任务
  GetExtendedAgentCard ：鉴权后获取扩展能力描述

流式事件不再使用 `kind` 和 `final`；通过成员名区分：

  {"statusUpdate": {"taskId": "...", "status": {...}}}
  {"artifactUpdate": {"taskId": "...", "artifact": {...}}}

终态由协议绑定的流关闭规则表示，而不是额外的 final 布尔值。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    """返回 A2A 示例使用的毫秒级 UTC 时间。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TaskState(str, Enum):
    SUBMITTED = "TASK_STATE_SUBMITTED"
    WORKING = "TASK_STATE_WORKING"
    COMPLETED = "TASK_STATE_COMPLETED"
    FAILED = "TASK_STATE_FAILED"
    CANCELED = "TASK_STATE_CANCELED"
    REJECTED = "TASK_STATE_REJECTED"
    INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
    AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"


TERMINAL_STATES = {
    TaskState.COMPLETED,
    TaskState.FAILED,
    TaskState.CANCELED,
    TaskState.REJECTED,
}


@dataclass(frozen=True)
class Caller:
    """网络层完成认证后传给业务层的调用者上下文。"""

    subject: str
    tenant: str
    scopes: frozenset[str] = frozenset()


@dataclass
class TaskRecord:
    id: str
    context_id: str
    owner_subject: str
    tenant: str
    status: TaskState
    created_at: str
    last_modified: str
    history: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def public_view(self, include_history: bool = False) -> dict[str, Any]:
        result = {
            "id": self.id,
            "contextId": self.context_id,
            "status": {"state": self.status.value, "timestamp": self.last_modified},
            "createdAt": self.created_at,
            "lastModified": self.last_modified,
            "artifacts": self.artifacts,
        }
        if include_history:
            result["history"] = self.history
        return result


class A2AError(Exception):
    pass


class A2ATeachingServer:
    """A2A v1.0 对象与操作的内存教学实现。

    真实服务应使用官方 SDK 生成 binding、google.rpc.Status 错误、分页 token、
    JWS 签名、OAuth/mTLS 与持久化。本类的关键点是 Task 隔离和 v1.0 命名。
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._message_index: dict[tuple[str, str, str], str] = {}

    @staticmethod
    def agent_card() -> dict[str, Any]:
        return {
            "name": "Agent Study Report Demo",
            "description": "使用模拟数据演示 A2A v1.0 对象模型",
            "version": "2026.8",
            "supportedInterfaces": [
                {
                    "url": "https://agents.example.invalid/report",
                    "protocolBinding": "JSONRPC",
                    "protocolVersion": "1.0",
                }
            ],
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "extendedAgentCard": False,
            },
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/markdown"],
            "skills": [
                {
                    "id": "course-summary",
                    "name": "课程摘要",
                    "description": "根据用户文本生成模拟摘要 Artifact",
                    "tags": ["summary", "education"],
                }
            ],
        }

    def send_message(self, caller: Caller, message: dict[str, Any]) -> dict[str, Any]:
        """实现 SendMessage 的教学子集，并用 messageId 保证重试幂等。"""
        self._require_scope(caller, "tasks:write")
        self._validate_message(message)
        message_id = message["messageId"]
        dedupe_key = (caller.tenant, caller.subject, message_id)
        existing_id = self._message_index.get(dedupe_key)
        if existing_id:
            return self._tasks[existing_id].public_view(include_history=True)

        now = utc_now()
        task_id = str(uuid.uuid4())
        context_id = str(message.get("contextId") or uuid.uuid4())
        task = TaskRecord(
            id=task_id,
            context_id=context_id,
            owner_subject=caller.subject,
            tenant=caller.tenant,
            status=TaskState.SUBMITTED,
            created_at=now,
            last_modified=now,
            history=[message],
        )
        self._tasks[task_id] = task
        self._message_index[dedupe_key] = task_id

        # 教学模拟为同步完成；真实 Server 可异步进入 WORKING 并流式发送事件。
        task.status = TaskState.WORKING
        task.last_modified = utc_now()
        user_text = " ".join(str(part.get("text", "")) for part in message["parts"])
        task.artifacts.append(
            {
                "artifactId": str(uuid.uuid4()),
                "name": "summary.md",
                "parts": [
                    {
                        "text": f"# 模拟摘要\n\n已接收：{user_text}",
                        "mediaType": "text/markdown",
                    }
                ],
            }
        )
        task.status = TaskState.COMPLETED
        task.last_modified = utc_now()
        return task.public_view(include_history=True)

    def get_task(self, caller: Caller, task_id: str, include_history: bool = False) -> dict[str, Any]:
        self._require_scope(caller, "tasks:read")
        return self._visible_task(caller, task_id).public_view(include_history)

    def list_tasks(self, caller: Caller, page_size: int = 20) -> dict[str, Any]:
        self._require_scope(caller, "tasks:read")
        bounded_size = max(1, min(page_size, 100))
        visible = [
            task.public_view(False)
            for task in self._tasks.values()
            if task.tenant == caller.tenant and task.owner_subject == caller.subject
        ]
        # 真实实现需返回不可伪造、稳定排序的 nextPageToken。
        return {"tasks": visible[:bounded_size], "nextPageToken": None}

    def cancel_task(self, caller: Caller, task_id: str) -> dict[str, Any]:
        self._require_scope(caller, "tasks:write")
        task = self._visible_task(caller, task_id)
        if task.status in TERMINAL_STATES:
            raise A2AError(f"task in terminal state: {task.status.value}")
        task.status = TaskState.CANCELED
        task.last_modified = utc_now()
        return task.public_view(False)

    def dispatch(self, caller: Caller, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        """用 v1.0 操作名模拟 binding 分派。"""
        if operation == "SendMessage":
            return self.send_message(caller, params["message"])
        if operation == "GetTask":
            return self.get_task(caller, params["id"], params.get("includeHistory", False))
        if operation == "ListTasks":
            return self.list_tasks(caller, params.get("pageSize", 20))
        if operation == "CancelTask":
            return self.cancel_task(caller, params["id"])
        raise A2AError(f"unsupported operation: {operation}")

    @staticmethod
    def _validate_message(message: dict[str, Any]) -> None:
        if message.get("role") != "ROLE_USER":
            raise A2AError("teaching server expects role ROLE_USER")
        if not isinstance(message.get("messageId"), str) or not message["messageId"]:
            raise A2AError("messageId is required")
        parts = message.get("parts")
        if not isinstance(parts, list) or not parts:
            raise A2AError("parts must be a non-empty list")
        for part in parts:
            content_members = {"text", "raw", "url", "data"}.intersection(part)
            if len(content_members) != 1:
                raise A2AError("each Part must contain exactly one content member")

    @staticmethod
    def _require_scope(caller: Caller, scope: str) -> None:
        if scope not in caller.scopes:
            raise A2AError(f"missing scope: {scope}")

    def _visible_task(self, caller: Caller, task_id: str) -> TaskRecord:
        task = self._tasks.get(task_id)
        if task is None or task.tenant != caller.tenant or task.owner_subject != caller.subject:
            # 不区分“不存在”和“无权访问”，避免枚举其他租户的 Task。
            raise A2AError("task not found")
        return task


"""
15.5 从 v0.x 迁移到 v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━

必须更新的高风险项：
  1. 操作名：message/send → SendMessage；tasks/get → GetTask；新增 ListTasks
  2. 枚举：小写/连字符 → ROLE_* 与 TASK_STATE_* 的 SCREAMING_SNAKE_CASE
  3. Part：删除 kind 和嵌套 file；使用 text/raw/url/data 成员判别
  4. AgentCard：迁移到 supportedInterfaces[]，协议版本下沉到 interface
  5. 流事件：删除 kind/final，使用 statusUpdate / artifactUpdate 成员
  6. 错误：按 binding 映射到标准化 google.rpc.Status 语义
  7. 安全：现代 OAuth、可选 mTLS、AgentCard JWS 签名与多租户 scope

推荐迁移流程：先建立 v0.x ↔ v1.0 兼容层与契约测试，再双栈灰度；所有调用方
升级完成后移除旧 binding。不要仅做字段搜索替换，Part 和事件判别逻辑必须重测。


15.6 生产检查清单
━━━━━━━━━━━━━━━━━━━

  - AgentCard 只声明稳定能力；验证签名、域名和安全方案
  - 每个 Task 绑定 authenticated subject 与 tenant，查询时再次授权
  - messageId / 业务幂等键防止网络重试产生重复副作用
  - 长任务设置超时、取消、续订、事件重放和 Artifact 保留策略
  - Push Notification 回调需验证目标、签名并防止 SSRF
  - 跨 Agent 输入和 Artifact 都是不可信内容，需扫描和最小化再进入模型上下文
  - 记录 operation、agent、task、tenant、状态转换、耗时和错误，但避免泄露正文
"""


def demo() -> None:
    server = A2ATeachingServer()
    caller = Caller(
        subject="user-42",
        tenant="tenant-demo",
        scopes=frozenset({"tasks:read", "tasks:write"}),
    )
    message = {
        "messageId": "msg-demo-001",
        "role": "ROLE_USER",
        "parts": [{"text": "请总结 A2A v1.0 的核心变化", "mediaType": "text/plain"}],
    }

    print("AgentCard v1.0:")
    print(json.dumps(server.agent_card(), ensure_ascii=False, indent=2))
    task = server.dispatch(caller, "SendMessage", {"message": message})
    print("\nSendMessage result:")
    print(json.dumps(task, ensure_ascii=False, indent=2))
    print("\nListTasks result:")
    print(json.dumps(server.dispatch(caller, "ListTasks", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()
