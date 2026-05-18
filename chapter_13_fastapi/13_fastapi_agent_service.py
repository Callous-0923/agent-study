"""
第13章：FastAPI + Agent 服务化部署
===================================

📌 本章目标：
  1. 用 FastAPI 将 Agent 封装为 RESTful API 服务
  2. 实现 SSE (Server-Sent Events) 流式推送 Agent 执行过程
  3. 实现 WebSocket 双向实时通信
  4. 掌握 Agent Webhook 回调模式
  5. 理解 Agent 微服务的部署架构（Docker / 负载均衡）
  6. 创建一个可直接运行的完整 Agent 服务

📌 面试高频点：
  - Agent 怎么做成 API 服务？有哪些关键设计决策？
  - SSE 和 WebSocket 有什么区别？Agent 场景分别怎么用？
  - 流式 Agent 的状态管理怎么做？
  - 生产环境的 Agent 服务架构是怎样的？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本章需要安装：pip install fastapi uvicorn pydantic
运行服务：python chapter_13_fastapi/13_fastapi_agent_service.py
测试地址：http://localhost:8000/docs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


13.1 为什么需要把 Agent 做成 API 服务？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

裸写 Agent 脚本的问题：
  - 只能命令行交互，无法集成到 Web/App
  - 每个用户需要单独启动进程
  - 没有并发的请求处理能力
  - 无法做认证、限流、监控

FastAPI 方案的优势：
  - ASGI 异步框架（天然支持 Agent 的异步调用）
  - 自动生成 OpenAPI 文档（/docs）
  - 内置 SSE / WebSocket 支持
  - 类型安全（Pydantic v2）
  - 异步并发（async/await 原生支持）
"""

import os
import json
import time
import sqlite3
import asyncio
import hashlib
from datetime import datetime
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
13.2 数据模型定义（Pydantic v2）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用 Pydantic v2 来定义 Agent 服务的请求和响应结构体。
选择 Pydantic 的理由：
  - FastAPI 原生集成，自动生成 OpenAPI / Swagger 文档
  - Field 描述直接变成 API 文档中的参数说明
  - 请求校验零代码——不合法的参数自动返回 422
  - 支持嵌套模型（Agent 的 tool_call 是嵌套结构）

这里定义两个核心模型：
  - AgentRequest：用户发来的任务请求
  - AgentStep：Agent 执行过程中每步的状态
"""


class AgentRequest(BaseModel):
    """Agent API 请求体。

    FastAPI + Pydantic 自动完成：
      1. JSON 解析
      2. 类型校验
      3. 生成 OpenAPI 文档
    """
    message: str = Field(..., min_length=1, max_length=10000,
                         description="用户输入的消息")
    session_id: Optional[str] = Field(None,
                                       description="会话ID，不传则自动创建")
    stream: bool = Field(False,
                         description="是否使用 SSE 流式返回")


class AgentResponse(BaseModel):
    """Agent API 响应体。"""
    session_id: str
    answer: str
    tool_calls_made: int
    elapsed_seconds: float
    timestamp: str


class SSEEvent(BaseModel):
    """SSE 事件格式。"""
    event: str                      # "thinking" | "tool_call" | "tool_result" | "done" | "error"
    data: str                       # JSON 字符串


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
13.3 模拟 Agent 引擎（不依赖外部 API）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这里实现一个 MockAgentEngine：不依赖任何外部 LLM API，用模拟步骤
来演示 Agent 的执行流程。为什么这样做？

  1. 可离线测试——不需要配置 API Key 也能跑通整个 HTTP 服务
  2. 展示架构分层——Agent 引擎和服务框架是解耦的
  3. 真实项目中的做法——先用 Mock 验证 API 设计，再接入真实 LLM

MockAgentEngine 的核心：
  - execute() 是异步的，返回 async generator（每步一个事件）
  - 模拟了「思考→计划→工具调用→总结」4 步流程
  - 用 asyncio.sleep() 模拟 LLM 调用的延迟，方便测试超时机制
"""


class SimulatedAgent:
    """模拟 Agent —— 演示 Agent 的逐步执行过程。

    实际项目中替换为 LangChain Agent / LangGraph / 自定义 Agent。
    这里模拟的思考-行动-观察循环，展示 SSE 推送的完整过程。
    """

    MOCK_TOOLS = {
        "search": lambda q: f"搜索结果: 关于「{q}」的最新信息...",
        "calculate": lambda e: f"计算结果: {eval(e) if all(c in '0123456789+-*/().% ' for c in e) else '表达式错误'}",
        "get_time": lambda _: f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    }

    async def run_streaming(
        self, message: str
    ) -> AsyncGenerator[dict, None]:
        """流式执行 Agent，每步通过 yield 推送状态。

        这是 SSE 的核心：不是等所有步骤完成再返回，
        而是每完成一步就推送一个事件。

        Args:
            message: 用户输入。

        Yields:
            每步的状态事件字典。
        """
        steps = []

        # 第1步：思考阶段
        yield {"event": "thinking", "data": "正在分析用户意图..."}
        await asyncio.sleep(0.3)

        # 判断需要哪些工具
        if "搜索" in message or "查" in message:
            keyword = message.replace("搜索", "").replace("查", "").strip() or "AI Agent"
            steps.append(("search", keyword))
        if "算" in message or "计算" in message or "+" in message or "*" in message:
            steps.append(("calculate", message))
        if "时间" in message or "几点" in message:
            steps.append(("get_time", ""))

        # 第2步：工具调用阶段
        for i, (tool_name, arg) in enumerate(steps):
            yield {
                "event": "tool_call",
                "data": json.dumps({"step": i+1, "tool": tool_name, "args": arg}, ensure_ascii=False),
            }
            await asyncio.sleep(0.2)

            result = self.MOCK_TOOLS[tool_name](arg)
            yield {
                "event": "tool_result",
                "data": json.dumps({"step": i+1, "tool": tool_name, "result": result}, ensure_ascii=False),
            }
            await asyncio.sleep(0.2)

        # 第3步：生成最终回答
        yield {"event": "thinking", "data": "正在组织最终回答..."}
        await asyncio.sleep(0.3)

        if steps:
            answer = f"已完成 {len(steps)} 个工具调用来回答您的问题。"
        else:
            answer = f"您好！您的问题「{message}」已收到。这是一个模拟 Agent 回复。"

        yield {"event": "done", "data": answer, "tool_calls_made": len(steps)}


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
13.4 FastAPI 应用主体
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是整个 Agent 服务的入口。FastAPI app 拆分为 6 个端点：

  POST /agent/task —— 提交一个 Agent 任务（同步等待结果）
  GET  /agent/task/{task_id}/stream —— SSE 流式获取执行过程
  WS   /agent/task/{task_id}/ws —— WebSocket 双向通信
  GET  /agent/task/{task_id} —— 查询任务状态
  GET  /health —— 健康检查
  GET  / —— Swagger 文档页面

加上 CORS 中间件（允许前端跨域调用）和应用生命周期管理。
"""

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的操作。"""
    print("🚀 Agent 服务启动中...")
    # 启动时：初始化数据库连接池、加载配置等
    yield
    print("👋 Agent 服务关闭中...")
    # 关闭时：清理资源、关闭连接等

app = FastAPI(
    title="AI Agent API Service",
    description="一个完整的 Agent 服务化部署示例",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件（允许前端跨域调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = SimulatedAgent()


# ==================== REST API 端点 ====================

@app.post("/agent/chat", response_model=AgentResponse)
async def agent_chat(request: AgentRequest):
    """标准 REST API：一次请求，一次完整响应。

    适用场景：
      - 简单问答（用户不关心中间过程）
      - 批量任务处理
      - 与其他微服务集成

    测试：
      curl -X POST http://localhost:8000/agent/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "搜索AI Agent并计算3+5"}'
    """
    start = time.time()
    session_id = request.session_id or _generate_session_id()
    tool_calls_made = 0
    answer = ""

    async for event in agent.run_streaming(request.message):
        if event["event"] == "done":
            answer = event["data"]
            tool_calls_made = event.get("tool_calls_made", 0)

    return AgentResponse(
        session_id=session_id,
        answer=answer,
        tool_calls_made=tool_calls_made,
        elapsed_seconds=round(time.time() - start, 2),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/agent/chat/stream")
async def agent_chat_stream(request: AgentRequest):
    """SSE 流式 API：逐步推送 Agent 的思考过程。

    适用场景：
      - 用户需要看到 Agent 的实时状态（「正在搜索...」）
      - 长任务（避免用户干等）
      - 前端需要展示工具调用动画

    测试：
      curl -N -X POST http://localhost:8000/agent/chat/stream \
        -H "Content-Type: application/json" \
        -d '{"message": "搜索AI Agent并算3+5"}'
    """

    async def event_generator():
        """SSE 事件生成器。

        SSE 协议格式：
          event: <事件类型>
          data: <JSON 数据>
          \n
        """
        async for event in agent.run_streaming(request.message):
            event_type = event["event"]
            data = event["data"]
            # SSE 标准格式
            yield f"event: {event_type}\ndata: {data}\n\n"

            if event_type == "done":
                yield f"event: close\ndata: stream_end\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx 禁用缓冲
        },
    )


# ==================== WebSocket 端点 ====================

@app.websocket("/agent/ws")
async def agent_websocket(websocket):
    """WebSocket 双向通信端点。

    与 SSE 的区别：
      - SSE：服务器 → 客户端（单向推送）
      - WebSocket：服务器 ↔ 客户端（双向通信）

    适用场景：
      - 用户可以在 Agent 执行中途发送消息（类似 Claude Code h2A）
      - 需要多次交互的对话
      - 实时协作场景

    测试：
      websocat ws://localhost:8000/agent/ws
    """
    await websocket.accept()
    await websocket.send_json({
        "event": "connected",
        "data": "WebSocket 连接已建立",
    })

    try:
        while True:
            # 接收用户消息
            raw = await websocket.receive_text()
            data = json.loads(raw)
            message = data.get("message", "")

            if not message:
                continue

            await websocket.send_json({
                "event": "received",
                "data": f"收到: {message}",
            })

            # 流式执行 Agent
            async for event in agent.run_streaming(message):
                await websocket.send_json(event)

            await websocket.send_json({
                "event": "ready",
                "data": "等待下一条消息...",
            })
    except Exception as e:
        print(f"WebSocket 断开: {e}")


# ==================== 管理端点 ====================

@app.get("/health")
async def health_check():
    """健康检查端点（Kubernetes / 负载均衡器）。"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/")
async def root():
    """根路径 → 跳转到 API 文档。"""
    return {
        "service": "AI Agent API Service",
        "docs": "/docs",
        "endpoints": {
            "POST /agent/chat": "标准 API（一次响应）",
            "POST /agent/chat/stream": "SSE 流式 API",
            "WebSocket /agent/ws": "WebSocket 双向通信",
            "GET /health": "健康检查",
        },
    }


# ==================== 辅助函数 ====================

def _generate_session_id() -> str:
    """生成唯一会话 ID。"""
    return hashlib.md5(
        f"{time.time()}-{os.urandom(8).hex()}".encode()
    ).hexdigest()[:16]


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
13.5 生产部署架构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────────────────────────────────────┐
  │                     Nginx (反向代理)                   │
  │  • SSL 终止                                           │
  │  • 负载均衡 (round-robin / least-connections)         │
  │  • Rate Limiting                                      │
  │  • WebSocket 升级 (Connection: Upgrade)               │
  └──────────┬───────────┬───────────┬───────────────────┘
             │           │           │
    ┌────────▼──┐ ┌──────▼───┐ ┌───▼────────┐
    │  Agent    │ │  Agent   │ │  Agent     │
    │  Worker 1 │ │  Worker 2│ │  Worker 3  │
    │ (FastAPI) │ │ (FastAPI)│ │  (FastAPI) │
    └───────────┘ └──────────┘ └────────────┘
             │           │           │
             └───────────┼───────────┘
                         │
              ┌──────────▼──────────┐
              │   Redis / RabbitMQ  │  ← 消息队列（异步任务）
              └─────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  SQLite / PostgreSQL│  ← 持久化存储（第14章详讲）
              └─────────────────────┘

Docker 部署示例：
  FROM python:3.12-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  EXPOSE 8000
  CMD ["uvicorn", "13_fastapi_agent_service:app", "--host", "0.0.0.0", "--port", "8000"]

Kubernetes 部署要点：
  - Readiness Probe: GET /health
  - Liveness Probe: GET /health
  - Horizontal Pod Autoscaler: CPU > 70% → 扩容
  - 使用 ConfigMap 管理 Agent 配置


13.6 SSE vs WebSocket vs 普通 HTTP —— 该怎么选？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────┬───────────────┬───────────────┬────────────────┐
│     特性      │  普通 HTTP     │     SSE        │   WebSocket    │
├──────────────┼───────────────┼───────────────┼────────────────┤
│ 通信方向      │ 请求→响应      │ 服务器→客户端   │ 双向           │
│ 协议          │ HTTP          │ HTTP           │ WS (升级HTTP)  │
│ 浏览器支持    │ ✅            │ ✅ (EventSource)│ ✅             │
│ 自动重连      │ 无            │ ✅ (内置)       │ 需手动实现      │
│ 二进制数据    │ ✅            │ ❌ (仅文本)     │ ✅             │
│ 穿透防火墙    │ ✅            │ ✅             │ 可能被拦截      │
│ Agent 场景    │ 简单问答       │ 流式输出        │ 实时双向对话    │
│ 实现复杂度    │ ⭐            │ ⭐⭐           │ ⭐⭐⭐          │
└──────────────┴───────────────┴───────────────┴────────────────┘

Agent 场景推荐：
  - 简单问答 → POST /agent/chat（普通 HTTP）
  - 流式输出（展示思考过程）→ POST /agent/chat/stream（SSE）
  - 实时对话（用户可中途介入）→ WebSocket /agent/ws


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
13.7 本章总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心要点回顾：

1. FastAPI 是最适合 Agent 的 Python Web 框架
   - 原生异步支持（Agent 的 LLM 调用就是异步的）
   - 自动文档生成
   - SSE / WebSocket 开箱即用

2. 三种通信模式各有适用场景
   - REST API: 简单问答 → 一次请求一次响应
   - SSE: 流式输出 → 展示 Agent 思考过程
   - WebSocket: 实时双向 → 用户可中途介入

3. 生产部署核心组件
   - Nginx: SSL + 负载均衡 + WebSocket 升级
   - 多 Worker: 水平扩展
   - 消息队列: 异步任务解耦
   - 健康检查: 自动故障恢复

面试速记：
  "Agent 怎么做成 API 服务？"
  → FastAPI + uvicorn 异步框架
  → SSE 做流式推送，WebSocket 做双向通信
  → Nginx 反代 + 多 Worker + 健康检查
"""


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第13章：FastAPI + Agent 服务化部署                    ║")
    print("║  REST API · SSE流式 · WebSocket · 生产架构           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("启动后访问以下地址：")
    print("  📖 API 文档:        http://localhost:8000/docs")
    print("  🏠 服务主页:        http://localhost:8000/")
    print("  ❤️  健康检查:       http://localhost:8000/health")
    print("  💬 REST 聊天:       POST http://localhost:8000/agent/chat")
    print("  📡 SSE 流式:        POST http://localhost:8000/agent/chat/stream")
    print("  🔌 WebSocket:       ws://localhost:8000/agent/ws")
    print()

    # 非阻塞启动（不调用外部 API，可独立运行）
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
