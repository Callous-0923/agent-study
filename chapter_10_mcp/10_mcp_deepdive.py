"""
第10章：MCP（Model Context Protocol）原理与调用全流程
=====================================================

📌 本章目标：
  1. 深入理解 MCP 协议的完整架构和设计哲学
  2. 掌握 JSON-RPC 2.0 消息格式和 MCP 的三种消息类型
  3. 理解 Client-Host-Server 三层架构
  4. 掌握 MCP 核心原语：Tools / Resources / Prompts / Sampling
  5. 学会 MCP 的生命周期管理（Initialize → Capability Negotiation → Teardown）
  6. 理解 stdio 和 SSE 两种传输方式的使用场景

📌 面试高频点：
  - MCP 和传统的 REST API 有什么区别？
  - MCP 的 Client-Host-Server 架构中，Host 的作用是什么？
  - 什么是 Capability Negotiation？怎么工作的？
  - MCP 的三个核心原语（Tools / Resources / Prompts）分别是什么？
  - MCP 和 Function Calling 是什么关系？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本章内容基于 MCP 官方规范 (2025-06-18 版) 和 SDK 源码
官方文档: https://modelcontextprotocol.io
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


10.1 MCP 是什么？—— 用类比建立直觉
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

类比：USB-C 接口

  在没有 USB-C 之前：
    手机用 Micro-USB、电脑用 USB-A、显示器用 HDMI、硬盘用 Thunderbolt
    每种设备需要不同的线，接口混乱不堪

  USB-C 统一了这一切：
    一个接口，连接所有设备
    协议统一，但每个设备的「功能」不同

  MCP 就是 AI 世界的 USB-C：
    一个协议，连接所有外部系统
    协议统一（JSON-RPC），但每个 Server 的「工具」不同
    2024年11月由 Anthropic 发布，迅速成为行业标准

MCP 解决了什么核心问题？

  旧世界（MCP 之前）：
    ┌──────┐   自定义协议   ┌──────────┐
    │ LLM   │──────────────→│ Google API│
    │       │   自定义协议   ├──────────┤
    │       │──────────────→│ GitHub API│
    │       │   自定义协议   ├──────────┤
    │       │──────────────→│ 数据库     │
    └──────┘                └──────────┘
    每连接一个新系统，都要写新的适配代码！

  新世界（MCP 之后）：
    ┌──────┐     MCP       ┌──────────┐
    │ LLM   │──────────────→│ MCP Server│→ Google
    │       │     MCP       ├──────────┤
    │       │──────────────→│ MCP Server│→ GitHub
    │       │     MCP       ├──────────┤
    │       │──────────────→│ MCP Server│→ Database
    └──────┘                └──────────┘
    一个协议，连接所有！新系统只需实现一个 MCP Server。


10.2 MCP 的三层架构 —— Host / Client / Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────────────────────────────────────────┐
  │                     Host Application                      │
  │  ┌─────────────────┐  ┌─────────────────┐                │
  │  │   MCP Client 1   │  │   MCP Client 2  │  ...          │
  │  │  (1:1 连接)     │  │  (1:1 连接)     │                │
  │  └────────┬────────┘  └────────┬────────┘                │
  │           │                    │                          │
  └───────────┼────────────────────┼──────────────────────────┘
              │                    │
     MCP协议  │           MCP协议  │
    (JSON-RPC)│          (JSON-RPC)│
              ▼                    ▼
  ┌──────────────────┐  ┌──────────────────┐
  │   MCP Server 1    │  │   MCP Server 2    │
  │   文件系统 & Git   │  │     数据库        │
  │                   │  │                   │
  │  • read_file()    │  │  • query()        │
  │  • write_file()   │  │  • insert()       │
  │  • git_status()   │  │  • schema()       │
  └──────────────────┘  └──────────────────┘

三个角色的职责：

  Host（主机）
    - LLM 所在的应用程序（如 Claude Desktop、Claude Code）
    - 创建和管理多个 Client 实例
    - 控制安全策略和用户授权
    - 聚合多个 Server 提供的上下文

  Client（客户端）
    - Host 内部的组件，每个 Client 连接一个 Server
    - 1:1 关系：一个 Client ↔ 一个 Server
    - 处理协议握手和能力协商
    - 路由协议消息

  Server（服务器）
    - 提供具体的上下文的进程/服务
    - 通过 MCP 原语暴露能力
    - 可以是本地进程（stdio）或远程服务（SSE）

关键设计原则（面试常问！）：
  1. Server 不能看到完整对话，也不能「看到」其他 Server
     → 安全隔离：对话历史留在 Host，Server 只收到必要信息
  2. Server 应极其简单易构建
     → 复杂逻辑留给 Host，Server 只做一件事
  3. Server 应高度可组合
     → 多个 Server 可以无缝组合使用


10.3 MCP 的消息格式 —— JSON-RPC 2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MCP 使用 JSON-RPC 2.0 作为通信协议。
这是理解 MCP 调用流程的基础。

三种消息类型：
"""

# === Request（请求）===
REQUEST_EXAMPLE = {
    "jsonrpc": "2.0",
    "id": 1,                    # 请求ID，用于匹配响应
    "method": "tools/call",     # 方法名
    "params": {                 # 参数
        "name": "get_weather",
        "arguments": {"city": "北京"}
    }
}

# === Response（成功响应）===
RESPONSE_EXAMPLE = {
    "jsonrpc": "2.0",
    "id": 1,                    # 对应请求的ID
    "result": {                 # 成功结果
        "content": [
            {"type": "text", "text": "北京天气：晴，25°C"}
        ]
    }
}

# === Error Response（错误响应）===
ERROR_RESPONSE_EXAMPLE = {
    "jsonrpc": "2.0",
    "id": 1,
    "error": {
        "code": -32602,         # 标准 JSON-RPC 错误码
        "message": "Invalid params",
        "data": {"detail": "缺少必填参数 'city'"}
    }
}

# === Notification（通知 — 无需响应）===
NOTIFICATION_EXAMPLE = {
    "jsonrpc": "2.0",
    "method": "notifications/tools/list_changed",  # 无 id 字段！
    "params": {}
}

"""
关键区别：
  - Request/Response 有 id 字段（双向匹配）
  - Notification 没有 id（不需要响应）


10.4 MCP 的核心方法（Primitives）—— 按调用流程
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

完整的 MCP 调用生命周期：

  ┌────────────────────────────────────────────┐
  │  Phase 1: 初始化 (Initialize)              │
  │  ┌──────────────────────────────────────┐  │
  │  │ Client                    Server     │  │
  │  │   │── initialize ──────────→│        │  │
  │  │   │←─ capabilities + info ─│        │  │
  │  │   │── initialized ─────────→│        │  │
  │  └──────────────────────────────────────┘  │
  ├────────────────────────────────────────────┤
  │  Phase 2: 发现工具 (Tool Discovery)         │
  │  ┌──────────────────────────────────────┐  │
  │  │   │── tools/list ──────────→│        │  │
  │  │   │←─ [工具1, 工具2, ...] ─│        │  │
  │  └──────────────────────────────────────┘  │
  ├────────────────────────────────────────────┤
  │  Phase 3: 调用工具 (Tool Invocation)        │
  │  ┌──────────────────────────────────────┐  │
  │  │   │── tools/call ──────────→│        │  │
  │  │   │←─ tool_result ────────│        │  │
  │  └──────────────────────────────────────┘  │
  ├────────────────────────────────────────────┤
  │  Phase 4: 关闭 (Teardown)                  │
  │  ┌──────────────────────────────────────┐  │
  │  │   │── 关闭连接 ─────────────→│        │  │
  │  └──────────────────────────────────────┘  │
  └────────────────────────────────────────────┘

MCP 的三大核心原语（Server 能力）：

  1. Tools（工具）
     - 让 LLM 可以「做」事情
     - 示例：查询数据库、调用API、执行计算
     - 对应方法：tools/list, tools/call

  2. Resources（资源）
     - 让 LLM 可以「读」数据
     - 示例：读取文件、获取数据库schema
     - 对应方法：resources/list, resources/read

  3. Prompts（提示模板）
     - 预定义的提示词模板
     - 示例：「代码审查模板」「Bug报告模板」
     - 对应方法：prompts/list, prompts/get

  4. Sampling（采样）—— Client 能力
     - Server 可以请求 Host 的 LLM 生成内容
     - 示例：Server 需要 LLM 帮助分类或总结


10.5 Capability Negotiation（能力协商）—— 面试重点！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MCP 不是「一刀切」的协议。Client 和 Server 在连接初始化时
互相声明自己的「能力」，只有双方都支持的功能才可用。

Server 声明的能力（capabilities）:
  {
    "capabilities": {
      "tools": {"listChanged": true},     // 支持工具，且工具列表可动态变化
      "resources": {"subscribe": true},   // 支持资源，且支持订阅更新
      "prompts": {"listChanged": false},  // 支持提示模板，列表静态不变
      "logging": {}                       // 支持日志输出
    }
  }

Client 声明的能力:
  {
    "capabilities": {
      "sampling": {},                     // 支持 LLM 采样
      "roots": {"listChanged": true},     // 支持根目录声明
      "experimental": {"featureX": {}}    // 实验性功能
    }
  }

为什么需要能力协商？
  1. 向后兼容：新版本可降级到老版本的功能集
  2. 渐进增强：功能可以逐步添加，不破坏现有 Server
  3. 安全隔离：Host 可以拒绝某些能力（如不启用 sampling）

  类比：两人见面先握手说「我会英语、中文、日语」
        然后选择双方都会的语言交流。


10.6 MCP 传输层 —— stdio vs SSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MCP 内置两种传输方式：

1. stdio（标准输入/输出）
   - 适用：本地进程、命令行工具
   - 原理：Client 启动 Server 作为子进程
   - 通信：通过 stdin/stdout 交换 JSON-RPC 消息
   - 优点：简单、安全（进程隔离）
   - 示例：本地文件系统 Server、本地数据库 Server

2. SSE（Server-Sent Events）
   - 适用：远程服务、Web 部署
   - 原理：HTTP POST（Client→Server）+ SSE（Server→Client）
   - 优点：支持远程部署、服务端推送
   - 示例：云端 API Server、第三方服务集成

还支持 2025 年新增的 Streamable HTTP 传输方式。
"""


class SimulatedMCPServer:
    """模拟 MCP Server —— 完整演示 MCP 的核心交互流程。

    实现了 Tools 原语的完整调用链路。
    """

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.capabilities = {
            "tools": {"listChanged": True},
            "resources": {"subscribe": False},
            "prompts": {"listChanged": False},
        }
        # 注册工具
        self.tools = {
            "get_weather": {
                "name": "get_weather",
                "description": "查询指定城市的天气信息",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"],
                },
                "handler": self._handle_weather,
            },
            "search_docs": {
                "name": "search_docs",
                "description": "在知识库中搜索文档",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
                "handler": self._handle_search,
            },
        }

    def _handle_weather(self, args: dict) -> dict:
        """处理天气查询工具调用。"""
        weather = {"北京": "晴 25°C", "上海": "多云 28°C",
                   "深圳": "阵雨 30°C"}
        city = args.get("city", "")
        return {
            "content": [{"type": "text",
                        "text": weather.get(city, f"未找到{city}的天气数据")}],
            "isError": city not in weather,
        }

    def _handle_search(self, args: dict) -> dict:
        """处理文档搜索工具调用。"""
        query = args.get("query", "")
        max_results = args.get("max_results", 5)
        mock_results = [f"文档{i}: 关于'{query}'的内容片段..." 
                        for i in range(min(max_results, 3))]
        return {
            "content": [{"type": "text", "text": "\n".join(mock_results)}],
            "isError": False,
        }

    def handle_request(self, request: dict) -> dict:
        """MCP Server 的消息路由（核心！）。

        Args:
            request: JSON-RPC 请求。

        Returns:
            JSON-RPC 响应。
        """
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        # === Phase 1: initialize ===
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": self.capabilities,
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version,
                    },
                },
            }

        # === Phase 2: tools/list ===
        if method == "tools/list":
            tool_list = []
            for t in self.tools.values():
                tool_list.append({
                    "name": t["name"],
                    "description": t["description"],
                    "inputSchema": t["inputSchema"],
                })
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": tool_list},
            }

        # === Phase 3: tools/call ===
        if method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            tool = self.tools.get(tool_name)
            if tool is None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"未知工具: {tool_name}"
                    },
                }
            result = tool["handler"](tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }

        # 未知方法
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"未知方法: {method}"
            },
        }


def demo_mcp_full_flow():
    """演示 MCP 的完整调用流程。"""
    print("=" * 60)
    print("  MCP 完整调用流程演示")
    print("=" * 60)

    server = SimulatedMCPServer("WeatherServer", "1.0.0")

    # Phase 1: 初始化 + 能力协商
    print("\n  [Phase 1] 初始化与能力协商")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"sampling": {}},
            "clientInfo": {"name": "ClaudeCode", "version": "1.0.60"},
        },
    }
    print(f"  Client → Server: {init_request['method']}")
    init_response = server.handle_request(init_request)
    caps = init_response["result"]["capabilities"]
    print(f"  Server → Client: 协议版本 {init_response['result']['protocolVersion']}")
    print(f"  Server 能力: {list(caps.keys())}")

    # 发送 initialized 通知
    print(f"  Client → Server: initialized (通知，无需响应)")

    # Phase 2: 发现工具
    print("\n  [Phase 2] 发现工具列表")
    list_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    print(f"  Client → Server: {list_request['method']}")
    list_response = server.handle_request(list_request)
    tools = list_response["result"]["tools"]
    for t in tools:
        print(f"    工具: {t['name']} - {t['description']}")

    # Phase 3: 调用工具
    print("\n  [Phase 3] 调用工具")
    call_requests = [
        ("tools/call", {"name": "get_weather", "arguments": {"city": "北京"}}),
        ("tools/call", {"name": "search_docs", "arguments": {"query": "MCP协议"}}),
        ("tools/call", {"name": "get_weather", "arguments": {"city": "火星"}}),
    ]
    for method, params in call_requests:
        req = {"jsonrpc": "2.0", "id": 3, "method": method, "params": params}
        print(f"  Client → Server: {params['name']}({params['arguments']})")
        response = server.handle_request(req)
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            is_error = response["result"].get("isError", False)
            flag = "⚠️ 错误:" if is_error else "  ✅"
            print(f"  Server → Client: {flag} {content}")
        else:
            print(f"  Server → Client: ❌ {response['error']['message']}")


"""
10.7 MCP vs OpenAI Function Calling —— 对比
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是面试中的高频问题！

  ┌──────────────┬──────────────────┬───────────────────┐
  │     维度      │   MCP             │  Function Calling  │
  ├──────────────┼──────────────────┼───────────────────┤
  │ 本质          │ 通信协议          │ LLM 能力            │
  │ 标准化        │ 开放标准          │ 厂商自有实现        │
  │ 连接模式      │ Client↔Server    │ 直接在 API 调用中    │
  │ 工具发现      │ tools/list 动态   │ 每次请求手动传入     │
  │ 服务器生态    │ 独立部署 MCP Server│ 无独立服务器概念     │
  │ 可组合性      │ 多 Server 组合    │ 依赖应用代码         │
  │ 跨模型支持    │ 是（协议层面）     │ 取决于厂商           │
  └──────────────┴──────────────────┴───────────────────┘

关系：
  - Function Calling 定义「LLM 如何决定调用工具」
  - MCP 定义「工具如何被发现、连接、执行」
  - 两者是互补的，不是替代关系

实际架构中的配合：
  1. MCP Client 通过 tools/list 获取所有工具定义
  2. 将工具定义转换为 OpenAI function 格式传给 LLM
  3. LLM 返回 function call
  4. MCP Client 通过 tools/call 执行工具
  5. 将结果返回给 LLM


10.8 本章总结
━━━━━━━━━━━━━━

核心要点回顾：

1. MCP = AI 世界的 USB-C
   - 一个协议连接所有外部系统
   - Client-Host-Server 三层架构
   - JSON-RPC 2.0 消息格式

2. 核心原语（面试必问！）
   - Tools：让 LLM「做事」
   - Resources：让 LLM「读数据」
   - Prompts：预定义提示模板
   - Sampling：Server 请求 LLM 生成

3. 能力协商
   - 初始化时双方声明 capabilities
   - 只有双方都支持的功能才可用
   - 确保向后兼容和渐进增强

4. 传输层
   - stdio：本地进程（简单、安全）
   - SSE：远程服务（HTTP + 推送）
   - Streamable HTTP：2025 新增

5. MCP vs Function Calling
   - MCP 是协议，FC 是 LLM 能力
   - 互补关系：MCP 负责连接管理，FC 负责执行决策

面试速记：
  "请解释 MCP 的工作原理"
  → 三层架构(Host/Client/Server) + JSON-RPC 通信
  → 生命周期：Initialize → Capability Negotiation → 
    Tools/Resources/Prompts → Teardown
  → 核心价值：标准化 LLM 与外部工具/数据的交互
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第10章：MCP 协议原理与调用全流程                      ║")
    print("║  JSON-RPC · 原语 · 能力协商 · stdio/SSE             ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n▶ 10.1-10.3 MCP 架构与消息格式")
    print("-" * 50)
    print("MCP = Client-Host-Server 架构")
    print("通信协议: JSON-RPC 2.0")
    print("消息类型: Request / Response / Notification")
    print()
    print("Request 示例:")
    import pprint
    pprint.pprint(REQUEST_EXAMPLE, width=60)
    print()
    print("Response 示例:")
    pprint.pprint(RESPONSE_EXAMPLE, width=60)

    print("\n▶ 10.6 MCP 完整调用流程演示")
    demo_mcp_full_flow()

    print("\n▶ 10.7 MCP vs Function Calling 对比")
    print("-" * 50)
    comparisons = [
        ("本质", "MCP: 通信协议", "FC: LLM 能力"),
        ("标准化", "MCP: 开放标准", "FC: 厂商自有实现"),
        ("工具发现", "MCP: tools/list 动态发现", "FC: 每次手动传入"),
        ("可组合性", "MCP: 多 Server 组合", "FC: 依赖应用代码"),
        ("关系", "MCP 管理连接", "FC 管理执行决策"),
    ]
    for dim, mcp, fc in comparisons:
        print(f"  {dim:12s}  {mcp:30s}  {fc}")

    print("\n✅ 第10章完成！")
