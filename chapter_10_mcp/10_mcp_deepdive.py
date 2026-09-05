"""
第10章：MCP 2026-07-28 —— 无状态工具与上下文协议
=================================================

内容核对：2026-08-01
说明：本章协议说明以 MCP 2026-07-28 规范为准；Python 类是教学模拟，
不替代官方 SDK，也不实现 HTTP、鉴权或进程管理。
官方变更说明：https://blog.modelcontextprotocol.io/posts/2026-07-28/

📌 本章目标：
  1. 理解 MCP 的 Host / Client / Server 分工
  2. 掌握 2026-07-28 无状态核心与早期会话式版本的区别
  3. 理解 tools / resources / prompts 三类 Server 原语
  4. 正确选择 stdio 与 Streamable HTTP
  5. 用 JSON-RPC 2.0 实现一个无状态教学 Server

📌 面试高频点：
  - MCP 解决什么问题，和普通 Function Calling 有何区别？
  - 为什么 2026-07-28 删除初始化握手与协议级会话？
  - stdio 与 Streamable HTTP 如何选？
  - 如何控制 MCP Server 的权限、供应链和输出污染风险？


10.1 MCP 的边界
━━━━━━━━━━━━━━━━━

MCP（Model Context Protocol）为 AI 应用连接外部能力提供标准消息协议：

  Host   ：面向用户的 AI 应用，负责权限、模型调用和整体生命周期
  Client ：Host 内连接某个 Server 的协议组件
  Server ：暴露 tools、resources、prompts 等能力

  ┌────────────── Host ──────────────┐
  │ Model / Policy / Approval / UI   │
  │      ┌────────┐  ┌────────┐      │
  │      │Client A│  │Client B│      │
  └──────┴───┬────┴───┬───────┴──────┘
             │ MCP     │ MCP
        ┌────▼────┐ ┌──▼──────┐
        │Files Srv│ │CRM Server│
        └─────────┘ └──────────┘

MCP 不等于 Agent 框架：它不替你规划任务、选择模型或决定是否批准危险操作。
它也不等于业务 API：Server 通常仍需在内部调用数据库、HTTP API 或本地程序。


10.2 2026-07-28 的关键变化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

早期 MCP 版本把 initialize / initialized、能力协商和协议级 session 作为
连接生命周期的一部分。2026-07-28 将核心改为无状态：

  - 删除初始化握手和协议级会话
  - 每个请求携带完成该请求所需的信息
  - 简化无服务器、负载均衡和水平扩展
  - roots、sampling、logging 等 Client 能力进入弃用路径
  - 核心传输聚焦 stdio 与 Streamable HTTP

同一版本还加入了几项容易被遗漏的工程能力：

  - HTTP 请求可用 Mcp-Method / Mcp-Name 等路由头，网关无需解析完整 body
  - server/discover 用于发现端点能力，降低客户端预配置耦合
  - list 响应可携带 ttlMs，客户端据此缓存 tools/resources/prompts 列表
  - Multi-Round-Trip Requests（MRTR）允许一次外层请求内完成多轮协议交互
  - Tasks、鉴权等能力通过扩展演进，不应再次塞回核心的隐藏会话状态

这些字段属于协议契约，部署时还要同步检查代理是否保留路由头、缓存是否按
ttlMs 失效，以及扩展是否被客户端和 Server 双方支持。

“无状态”不表示业务不能保存状态。购物车、数据库事务或长任务仍可由业务层
用显式资源 ID、任务 ID 或外部存储管理；不要重新发明隐藏的粘性会话。


10.3 Server 原语
━━━━━━━━━━━━━━━━

┌───────────┬──────────────────────────────┬────────────────────────┐
│ 原语      │ 常见操作                     │ 责任边界               │
├───────────┼──────────────────────────────┼────────────────────────┤
│ tools     │ tools/list, tools/call       │ 可执行动作；Host 决定批准│
│ resources │ resources/list, resources/read│ 可寻址的上下文数据       │
│ prompts   │ prompts/list, prompts/get    │ 参数化提示模板           │
└───────────┴──────────────────────────────┴────────────────────────┘

工具 schema 是契约，不是安全边界。Server 仍需验证参数、授权调用者、限制资源，
Host 仍需根据副作用和数据敏感度决定自动执行、人工确认或拒绝。


10.4 传输层选型
━━━━━━━━━━━━━━━━

stdio：
  - Host 启动本地 Server 子进程，通过 stdin/stdout 交换 JSON-RPC
  - 适合桌面工具、CLI 和单用户本地集成
  - stdout 只能写协议消息；日志应写 stderr

Streamable HTTP：
  - Client 向单一 MCP 端点发送 HTTP POST
  - Server 可直接返回 JSON，也可用 SSE 流式返回多个事件
  - 适合远程、多实例和网关部署
  - 必须处理鉴权、Origin 校验、TLS、限流和 SSRF/重放风险

旧的独立 HTTP+SSE 会话传输只适合维护旧客户端，不应作为新系统默认方案。


10.5 JSON-RPC 2.0 教学实现
━━━━━━━━━━━━━━━━━━━━━━━━━━━

下面的 `StatelessMCPServer` 只演示协议分派和 schema。它没有网络监听、
OAuth、动态工具发现或完整错误码映射。每个 `handle()` 调用都可独立完成，
代码中没有 initialize 状态或 session 字段。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


JSONRPC_VERSION = "2.0"
MCP_SPEC_SNAPSHOT = "2026-07-28"


class MCPError(Exception):
    """带 JSON-RPC 错误码的教学异常。"""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True)
class Tool:
    """Server 暴露的一个函数工具。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    read_only: bool = True

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {"readOnlyHint": self.read_only},
        }


class StatelessMCPServer:
    """无状态 MCP 教学 Server。

    真实项目应使用与目标规范兼容的官方 SDK，并在传输层接入身份认证、
    限流、审计和超时。本类只实现三类 Server 原语的最小子集。
    """

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self._tools: dict[str, Tool] = {}
        self._resources: dict[str, str] = {}
        self._prompts: dict[str, str] = {}

    def add_tool(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def add_resource(self, uri: str, text: str) -> None:
        self._resources[uri] = text

    def add_prompt(self, name: str, template: str) -> None:
        self._prompts[name] = template

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """处理一个独立 JSON-RPC 请求；notification 不返回响应。"""
        request_id = request.get("id")
        try:
            self._validate_envelope(request)
            method = request["method"]
            params = request.get("params") or {}
            result = self._dispatch(method, params)
            if request_id is None:
                return None
            return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}
        except MCPError as exc:
            if request_id is None:
                return None
            error = {"code": exc.code, "message": exc.message}
            if exc.data is not None:
                error["data"] = exc.data
            return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}

    @staticmethod
    def _validate_envelope(request: dict[str, Any]) -> None:
        if request.get("jsonrpc") != JSONRPC_VERSION:
            raise MCPError(-32600, "Invalid Request: jsonrpc must be '2.0'")
        if not isinstance(request.get("method"), str):
            raise MCPError(-32600, "Invalid Request: method must be a string")
        if "params" in request and not isinstance(request["params"], dict):
            raise MCPError(-32602, "Invalid params: expected object")

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "tools/list": self._list_tools,
            "tools/call": self._call_tool,
            "resources/list": self._list_resources,
            "resources/read": self._read_resource,
            "prompts/list": self._list_prompts,
            "prompts/get": self._get_prompt,
        }
        handler = handlers.get(method)
        if handler is None:
            raise MCPError(-32601, f"Method not found: {method}")
        return handler(params)

    def _list_tools(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"tools": [tool.descriptor() for tool in self._tools.values()]}

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        tool = self._tools.get(name)
        if tool is None:
            raise MCPError(-32602, f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise MCPError(-32602, "arguments must be an object")
        try:
            value = tool.handler(**arguments)
            return {
                "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
                "isError": False,
            }
        except (TypeError, ValueError) as exc:
            return {
                "content": [{"type": "text", "text": f"tool input error: {exc}"}],
                "isError": True,
            }

    def _list_resources(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "resources": [
                {"uri": uri, "name": uri.rsplit("/", 1)[-1], "mimeType": "text/plain"}
                for uri in self._resources
            ]
        }

    def _read_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        if uri not in self._resources:
            raise MCPError(-32602, f"Unknown resource: {uri}")
        return {
            "contents": [
                {"uri": uri, "mimeType": "text/plain", "text": self._resources[uri]}
            ]
        }

    def _list_prompts(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompts": [
                {
                    "name": name,
                    "description": "教学模板",
                    "arguments": [{"name": "topic", "required": True}],
                }
                for name in self._prompts
            ]
        }

    def _get_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        template = self._prompts.get(name)
        if template is None:
            raise MCPError(-32602, f"Unknown prompt: {name}")
        topic = str((params.get("arguments") or {}).get("topic", ""))
        return {
            "description": "渲染后的教学模板",
            "messages": [
                {"role": "user", "content": {"type": "text", "text": template.format(topic=topic)}}
            ],
        }


def build_demo_server() -> StatelessMCPServer:
    """创建一个只含模拟数据的 Server。"""
    server = StatelessMCPServer("agent-study-demo", "2026.8")
    server.add_tool(
        Tool(
            name="get_weather",
            description="返回课程内置的模拟天气，不访问真实天气服务。",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            handler=lambda city: {"city": city, "condition": "晴（模拟）", "celsius": 25},
        )
    )
    server.add_resource("course://mcp/checklist", "schema、鉴权、超时、审批、审计、限流")
    server.add_prompt("review", "请审查 {topic} 的权限、输入验证和失败恢复策略。")
    return server


"""
10.6 生产安全清单
━━━━━━━━━━━━━━━━━━━

1. 身份与授权
   - 远程 Server 验证访问令牌的 audience、scope、过期时间和签名
   - 每个工具按调用者和资源做授权，不把“能看到工具”当作“能执行工具”

2. 最小权限与审批
   - 只读、可逆写入、不可逆操作分级
   - 删除、转账、发布、发送消息等高影响操作要求明确确认

3. 输入与输出
   - JSON Schema 之外再做长度、路径、域名、枚举和业务约束
   - 把工具输出视为不可信数据，防止间接提示注入进入模型上下文

4. 传输与运行时
   - 本地 HTTP 校验 Origin 并优先绑定 loopback
   - 设置超时、并发上限、输出上限、速率限制和熔断
   - 固定 Server/依赖版本，维护工具清单和供应链来源

5. 可观测与恢复
   - 记录调用者、工具名、参数摘要、审批、耗时和结果状态
   - 使用幂等键、补偿事务或人工恢复处理副作用


10.7 与相邻协议的关系
━━━━━━━━━━━━━━━━━━━━━━

  Function Calling：模型输出“调用哪个函数及参数”的供应商 API 机制
  MCP             ：Host/Client/Server 之间发现和调用外部能力的开放协议
  A2A             ：独立 Agent 服务之间发现、委派、跟踪任务和交换产物

常见组合：模型通过 Function Calling 选择工具，Host 通过 MCP Client 调用工具；
当任务需要委派给另一个独立 Agent 服务时，再通过 A2A 交换 Message/Task。
"""


def demo() -> None:
    server = build_demo_server()
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "get_weather", "arguments": {"city": "上海"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/read",
            "params": {"uri": "course://mcp/checklist"},
        },
    ]
    print(f"MCP 规范快照：{MCP_SPEC_SNAPSHOT}（无状态教学模拟）")
    for request in requests:
        print(json.dumps(server.handle(request), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()
