"""
第4章：主流 Agent 框架实战
===========================

📌 本章目标：
  1. 了解当前主流 Agent 框架及其定位
  2. 掌握 LangChain Agent 的核心用法
  3. 理解 LangGraph 的状态机模型
  4. 学会根据项目需求选择合适的框架

📌 面试高频点：
  - LangChain Agent vs 裸写的区别和优势
  - LangGraph 的状态管理机制
  - 生产环境 Agent 选型策略
  - 不同框架的适用场景


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4.0 主流框架全景图
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────────┬──────────────┬────────────────────────────┐
│       框架          │     定位      │          核心特点           │
├────────────────────┼──────────────┼────────────────────────────┤
│ LangChain          │ 通用 LLM 框架 │ 组件化、生态最丰富          │
│ LangGraph          │ Agent 编排    │ 状态机模型、图式工作流      │
│ crewAI             │ 多智能体协作  │ 角色扮演、任务委派           │
│ AutoGen (微软)      │ 对话式 Agent  │ 多Agent对话、人在回路        │
│ OpenAI Assistants  │ 托管式 Agent  │ 零配置、但灵活性低           │
│ Dify / Coze        │ 低代码 Agent  │ 可视化搭建、非技术人员友好   │
└────────────────────┴──────────────┴────────────────────────────┘

框架选择速查表（面试高频！）：
┌──────────────────────────────────────┬────────────────────────┐
│                   场景                │        推荐框架         │
├──────────────────────────────────────┼────────────────────────┤
│ 快速原型、学习研究                    │ LangChain              │
│ 复杂工作流、生产环境                  │ LangGraph              │
│ 多角色协作（市场分析、内容创作）       │ crewAI                 │
│ 需要人在回路的对话系统                │ AutoGen                │
│ 快速搭建内部工具                      │ Dify / Coze            │
│ 完全掌控、极致性能                    │ 裸写（第1章的方式）     │
└──────────────────────────────────────┴────────────────────────┘

下面我们重点学习 LangChain Agent 和 LangGraph。
这两个是目前市场上使用最广泛的框架。
"""

import os
import json
from typing import Any
from dotenv import load_dotenv

load_dotenv()

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4.1 LangChain Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LangChain 是最流行的 LLM 应用框架，它将第1-3章我们
手写的概念（Agent、Tool、Memory）都抽象为了标准组件。

核心组件映射：
  手写版          →  LangChain 版
  ─────────────    ─────────────────
  TOOLS 列表       →  @tool 装饰器
  TOOL_MAP 字典    →  Tool 对象
  Agent 循环       →  create_react_agent()
  对话历史         →  MemorySaver / ChatMessageHistory

安装依赖:
  pip install langchain langchain-openai langgraph
"""

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain 未安装。请运行: pip install langchain langchain-openai langgraph")


@tool
def get_stock_price(symbol: str) -> str:
    """查询股票实时价格。

    Args:
        symbol: 股票代码，如 AAPL（苹果）、TSLA（特斯拉）、BABA（阿里巴巴）。

    Returns:
        该股票的最新价格信息。
    """
    mock_prices = {
        "AAPL": "185.50 USD",
        "TSLA": "245.30 USD",
        "GOOGL": "175.20 USD",
        "MSFT": "420.10 USD",
        "BABA": "85.50 USD",
    }
    price = mock_prices.get(symbol.upper(), f"未找到股票 {symbol} 的数据")
    return f"{symbol.upper()}: {price} (模拟数据)"


@tool
def search_news(query: str) -> str:
    """搜索最新的新闻资讯。

    适用于获取实时新闻、事件报道、行业动态等。

    Args:
        query: 搜索关键词，如 "AI 行业动态"。

    Returns:
        相关新闻摘要。
    """
    mock_news = {
        "AI": "2025年 AI Agent 成为业界焦点，各大科技公司纷纷布局 Agent 技术。",
        "股票": "今日股市震荡，科技板块表现强劲。",
        "特斯拉": "特斯拉发布新一代自动驾驶技术，股价上涨 3%。",
    }
    for key in mock_news:
        if key in query:
            return mock_news[key]
    return f"关于 '{query}' 的新闻：行业持续发展，暂无重大事件。(模拟数据)"


@tool
def calculate_math(expression: str) -> str:
    """执行数学计算。

    Args:
        expression: 数学表达式，如 '100 * 1.15' 或 'sqrt(144)'。

    Returns:
        计算结果。
    """
    try:
        allowed = set("0123456789+-*/().%^ ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符"
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


def demo_langchain_agent():
    """演示 LangChain Agent 的基本用法。

    对比第1章的裸写 Agent，可以看到：
      1. 工具定义用 @tool 装饰器，更简洁
      2. Agent 执行器一行代码创建
      3. 对话历史自动管理（MemorySaver）
      4. 线程级别的对话隔离
    """
    if not LANGCHAIN_AVAILABLE:
        print("需要安装 LangChain 才能运行此演示。")
        return

    print("\n" + "=" * 60)
    print("  LangChain Agent 演示")
    print("=" * 60)

    # 步骤 1：初始化 LLM
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.7,
    )

    # 步骤 2：准备工具列表
    tools = [get_stock_price, search_news, calculate_math]

    # 步骤 3：创建 Agent（内置 ReAct 循环！）
    memory = MemorySaver()  # 对话记忆管理器
    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
    )

    """
    LangChain Agent 的核心抽象：
      - LLM: ChatOpenAI —— 大脑
      - Tools: @tool 列表 —— 技能
      - Agent: create_react_agent —— 执行器
      - Checkpointer: MemorySaver —— 记忆

    对比第1章裸写的优势：
      1. 不需要手动管理 messages 列表
      2. 不需要手动处理 tool_calls 循环
      3. 对话历史自动持久化
      4. 支持多线程（通过 thread_id 隔离对话）
    """

    # 步骤 4：运行对话
    test_queries = [
        "帮我查一下苹果(AAPL)和特斯拉(TSLA)的股价",
        "特斯拉最近有什么新闻吗？",
    ]

    config = {"configurable": {"thread_id": "demo-session-1"}}

    for query in test_queries:
        print(f"\n👤 用户: {query}")

        result = agent.invoke(
            {"messages": [("user", query)]},
            config=config,
        )

        # 最后一条消息是 Agent 的回复
        final_msg = result["messages"][-1]
        print(f"🤖 Agent: {final_msg.content}")


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4.2 LangGraph —— 状态机 Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LangGraph 是 LangChain 生态中的 Agent 编排框架。
它的核心思想是用「有向图」来定义 Agent 的执行流程。

核心概念：
  - State: 贯穿整个流程的共享状态（TypedDict）
  - Node: 图中的一个执行节点（Python 函数）
  - Edge: 节点之间的连线（普通边 / 条件边）

什么时候用 LangGraph 而不是 LangChain Agent？
  - 需要自定义执行流程（不是简单的 ReAct 循环）
  - 需要在步骤间传递复杂状态
  - 需要并行执行、条件分支、循环等复杂控制流
  - 多 Agent 协作场景（第5章会用到）

安装依赖：
  pip install langgraph
"""

try:
    from typing import Annotated, TypedDict
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph 未安装。")


# 定义状态类型
class AgentState(TypedDict):
    """LangGraph Agent 的状态定义。

    这是 LangGraph 最核心的概念。所有节点共享这个状态，
    节点通过读取/修改状态来协作完成任务。

    Annotated[list, add_messages] 表示消息列表使用
    追加模式（新消息添加到末尾，不会覆盖）。
    """
    messages: Annotated[list, add_messages]
    task_result: str
    step_count: int


def demo_langgraph_agent():
    """演示 LangGraph 状态机 Agent。

    实现一个简单的分析流程：
      输入问题 → LLM 分析 → 判断是否需要工具 → 输出结论

    LangGraph 让我们可以自定义 Agent 的执行图结构，
    而不是拘泥于固定的 ReAct 循环。
    """
    if not LANGGRAPH_AVAILABLE or not LANGCHAIN_AVAILABLE:
        print("需要安装 langgraph 和 langchain 才能运行此演示。")
        return

    print("\n" + "=" * 60)
    print("  LangGraph 状态机 Agent 演示")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.5,
    )

    # ===== 定义节点函数 =====
    def analyze_node(state: AgentState) -> AgentState:
        """分析节点：让 LLM 理解用户输入。"""
        print("  [Node: analyze] 正在分析用户输入...")
        response = llm.invoke([
            ("system", "分析用户问题，提取关键信息。简洁回答。"),
            ("user", state["messages"][-1].content),
        ])
        state["messages"].append(response)
        state["task_result"] = response.content
        state["step_count"] = state.get("step_count", 0) + 1
        return state

    def should_continue(state: AgentState) -> str:
        """条件判断：是否需要进一步处理？"""
        result = state["task_result"]
        if len(result) < 50:
            print("  [Edge] 回答太短，需要补充详细分析")
            return "elaborate"
        print("  [Edge] 回答充分，结束流程")
        return "end"

    def elaborate_node(state: AgentState) -> AgentState:
        """扩展节点：补充详细分析。"""
        print("  [Node: elaborate] 正在补充详细分析...")
        response = llm.invoke([
            ("system", "请更详细地展开分析，提供更多上下文和细节。"),
            ("user", state["task_result"]),
        ])
        state["messages"].append(response)
        state["task_result"] = response.content
        state["step_count"] = state.get("step_count", 0) + 1
        return state

    # ===== 构建图 =====
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("elaborate", elaborate_node)

    # 设置入口点
    workflow.set_entry_point("analyze")

    # 添加条件边：从 analyze 根据条件走向不同分支
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "elaborate": "elaborate",
            "end": END,
        },
    )

    # elaborate 完成后结束
    workflow.add_edge("elaborate", END)

    # 编译图
    app = workflow.compile()

    """
    图结构示意：

      [START]
         │
         ▼
    ┌──────────┐     条件判断      ┌───────────┐
    │  analyze  ├──────────────────→│ elaborate  │
    └─────┬─────┘  (太短)          └─────┬─────┘
          │ (足够)                       │
          ▼                              ▼
       [END]                          [END]

    这就是 LangGraph 的核心优势：
    用代码定义复杂的执行流程，而非硬编码在循环中。
    """

    # 运行
    test_input = "LangGraph 是什么？请简短说明。"
    print(f"\n👤 用户: {test_input}")

    initial_state = {
        "messages": [HumanMessage(content=test_input)],
        "task_result": "",
        "step_count": 0,
    }

    final_state = app.invoke(initial_state)
    print(f"\n🤖 Agent: {final_state['task_result']}")
    print(f"   总步骤数: {final_state['step_count']}")


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4.3 框架选择策略（面试高频！）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

面试官：「你用过哪些 Agent 框架？怎么选型？」

回答框架：

1. 学习阶段：从裸写开始（第1章）
   - 理解底层原理
   - 不被框架黑盒限制

2. 原型验证：用 LangChain Agent
   - 快速搭建、生态丰富
   - 适合 MVP 和 demo

3. 生产环境：用 LangGraph
   - 状态管理可控
   - 支持复杂工作流
   - 可观测性好（便于调试和监控）

4. 多 Agent 场景：
   - crewAI（简单协作）
   - AutoGen（需要人在回路）
   - 自建 LangGraph 图（完全定制）

选型核心考虑因素：
  ┌──────────────┬────────────────────────────────────┐
  │    因素       │               说明                  │
  ├──────────────┼────────────────────────────────────┤
  │ 复杂度        │ 简单用 LangChain，复杂用 LangGraph  │
  │ 可控性        │ 生产环境优先 LangGraph              │
  │ 团队能力      │ 新手用 LangChain，熟手自建           │
  │ 成本          │ 裸写 < LangChain < LangGraph       │
  │ 生态          │ LangChain 生态最丰富                │
  └──────────────┴────────────────────────────────────┘
"""


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4.4 本章总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心要点回顾：

1. LangChain Agent
   - create_react_agent() 一行创建 Agent
   - @tool 装饰器定义工具
   - MemorySaver 自动管理对话历史
   - 适合快速原型和中等复杂度场景

2. LangGraph
   - 状态机模型：State + Node + Edge
   - 完全自定义执行流程
   - 支持条件分支、循环、并行
   - 生产环境首选

3. 选型策略
   - 学习原理 → 裸写
   - 快速验证 → LangChain
   - 生产落地 → LangGraph
   - 多Agent → crewAI / AutoGen / 自建

面试速记：
  "LangGraph 和 LangChain Agent 的区别？"
  → LangChain Agent 是预定义的 ReAct 循环
  → LangGraph 可以自定义任意执行图
  → LangGraph 适合复杂工作流，LangChain Agent 适合简单任务
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║       第4章：主流 Agent 框架实战                       ║")
    print("║       LangChain · LangGraph · 选型策略               ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n▶ 4.1 LangChain Agent 演示")
    try:
        demo_langchain_agent()
    except Exception as e:
        print(f"  ⚠️ 运行出错（请检查依赖和 API Key）: {e}")

    print("\n▶ 4.2 LangGraph 状态机演示")
    try:
        demo_langgraph_agent()
    except Exception as e:
        print(f"  ⚠️ 运行出错（请检查依赖和 API Key）: {e}")

    print("\n▶ 4.3 框架选型速查")
    print("-" * 50)
    print("场景: 快速原型 → LangChain Agent")
    print("场景: 复杂工作流 → LangGraph")
    print("场景: 多Agent协作 → crewAI / 自建 LangGraph 图")
    print("场景: 人在回路 → AutoGen")
    print("场景: 完全掌控 → 裸写（第1章方式）")

    print("\n✅ 第4章完成！接下来进入第5章：多智能体系统。")
