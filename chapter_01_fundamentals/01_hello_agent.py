"""
第1章：AI Agent 基础概念 —— 从 LLM 调用到第一个 Agent
=====================================================

📌 本章目标：
  1. 理解 Agent 的 4 要素：LLM / 规划 / 记忆 / 工具
  2. 掌握 LLM API 调用的基本方式
  3. 手写一个最简 Agent（无框架，理解底层原理）
  4. 运行本文件，见到你的第一个 Agent 输出

📌 面试高频点：
  - Agent 循环（Perceive → Think → Act → Observe）的执行流程
  - Agent 和普通 LLM 调用的本质区别
  - 什么是 Function Calling，为什么它是 Agent 的基石


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.1 从最基础的 LLM 调用开始
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


def call_llm(prompt: str) -> str:
    """最基础的 LLM 调用 —— 发送提示词，获取回复。

    这是 Agent 的最底层操作：让 LLM「思考」。
    后续所有 Agent 逻辑都建立在这个基础之上。

    Args:
        prompt: 用户输入的提示词文本。

    Returns:
        LLM 生成的回复文本。
    """
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.2 Function Calling —— Agent 的「手」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Function Calling 是 Agent 的核心机制。LLM 不只是生成文本，
它还能「决定」调用哪个函数、传什么参数。

流程：
  用户：「北京今天天气怎么样？」
  →
  LLM 分析 → 识别需要调用 get_weather 函数
  →
  LLM 输出: {"name": "get_weather", "arguments": {"city": "北京"}}
  →
  我们的代码执行这个函数，拿到真实数据
  →
  把结果返回给 LLM，LLM 组织成自然语言回答用户

类比：LLM 像一位「盲人指挥家」，他能决定需要什么信息，
但必须由我们（代码）去执行实际的工具调用。
"""

# 定义工具列表 —— 告诉 LLM 它有哪些"技能"可以使用
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'上海'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "在网上搜索信息，返回搜索结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2 + 3 * 4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def get_weather(city: str) -> str:
    """模拟天气查询工具。

    Args:
        city: 城市名称。

    Returns:
        模拟的天气信息字符串。
    """
    weather_data = {
        "北京": "晴天，25°C，湿度 40%，北风 3 级",
        "上海": "多云，28°C，湿度 65%，东南风 2 级",
        "深圳": "阵雨，30°C，湿度 80%，南风 4 级",
        "成都": "阴天，22°C，湿度 55%，无持续风向",
    }
    return weather_data.get(city, f"未找到 {city} 的天气数据")


def search_web(query: str) -> str:
    """模拟网络搜索工具。

    Args:
        query: 搜索关键词。

    Returns:
        模拟的搜索结果。 字典形式，键为搜索关键词，值为搜索结果。
        如果没有找到相关结果，返回空字符串。
       """
    mock_results = {
        "python": "Python 是一种解释型、面向对象的高级编程语言。",
        "agent": (
            "AI Agent 是一种能够自主感知环境、做出决策并执行行动的"
            "智能系统。它通常由 LLM、规划器、记忆系统和工具组成。"
        ),
        "langchain": (
            "LangChain 是一个用于构建 LLM 应用的开源框架，"
            "提供了 Agent、Chain、Tool 等核心抽象。"
        ),
    }
    for key in mock_results:
        if key in query.lower():
            return mock_results[key]
    return f"关于 '{query}' 的搜索结果：这是一个模拟结果。"


def calculate(expression: str) -> str:
    """安全的数学计算工具。

    Args:
        expression: 数学表达式字符串。

    Returns:
        计算结果字符串。
    """
    try:
        allowed_chars = set("0123456789+-*/().%^ ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


# 工具名称到实际函数的映射
TOOL_MAP = {
    "get_weather": get_weather,
    "search_web": search_web,
    "calculate": calculate,
}


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.3 手写 Agent 循环 —— 理解底层原理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是本章最核心的部分！

每个 Agent 的运行本质上是一个循环：
  1. 把用户消息发给 LLM
  2. LLM 判断：直接回答？还是需要调用工具？
  3. 如果需要调用工具 → 执行工具 → 把结果发给 LLM，回到步骤 1
  4. 如果可以回答 → 输出答案，结束

这个循环被称为 ReAct 循环（Reasoning + Acting），
是 LangChain / AutoGPT / crewAI 等所有框架的底层逻辑。
"""


def run_agent(user_message: str, max_iterations: int = 5):
    """运行一个最简单的 ReAct Agent。

    这是不加任何框架的「裸写」Agent，帮助理解底层原理。
    任何 Agent 框架（LangChain/crewAI/AutoGen）本质上
    都是这个循环的封装 + 增强。

    Args:
        user_message: 用户输入的消息。
        max_iterations: 最大工具调用轮数，防止无限循环。

    Returns:
        Agent 的最终回答。
    """
    # 对话历史 —— Agent 的「短期记忆」
    messages = [
        {"role": "system", "content": "你是一个有用的 AI 助手。当需要获取最新信息或执行计算时，请使用提供的工具。"},
        {"role": "user", "content": user_message},
    ]

    print(f"\n{'='*60}")
    print(f"用户: {user_message}")
    print(f"{'='*60}")

    for iteration in range(max_iterations):
        # ===== 第 1 步：调用 LLM（Think 阶段）=====
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        # ===== 第 2 步：检查 LLM 是否需要调用工具 =====
        if assistant_message.tool_calls is None:
            # LLM 决定直接回答（不需要工具）→ 循环结束
            final_answer = assistant_message.content
            print(f"\n🎯 Agent 最终回答:\n{final_answer}")
            return final_answer

        # ===== 第 3 步：执行工具调用（Act 阶段）=====
        # 将 LLM 的回复（含 tool_calls）加入对话历史
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"\n🔧 第 {iteration + 1} 轮 - 调用工具: {func_name}({func_args})")

            # 执行实际的工具函数
            tool_function = TOOL_MAP.get(func_name)
            if tool_function is None:
                result = f"错误：未找到工具 {func_name}"
            else:
                result = tool_function(**func_args)

            print(f"📊 工具返回: {result}")

            # 将工具执行结果加入对话历史（Observe 阶段）
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # 达到最大迭代次数，强制 LLM 给出最终回答
    print("\n⚠️ 达到最大迭代次数，要求 LLM 给出最终回答...")
    messages.append({
        "role": "user",
        "content": "请基于已有的工具调用结果，给出最终回答。"
    })
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    final_answer = final_response.choices[0].message.content
    print(f"\n🎯 Agent 最终回答:\n{final_answer}")
    return final_answer


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.4 运行测试 —— 见证你的第一个 Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

下面准备了几组测试用例，分别测试：
  - Test 1: 不需要工具的简单问答（LLM 直接回答）
  - Test 2: 单工具调用（天气查询）
  - Test 3: 多工具调用（搜索 + 计算）
  - Test 4: 需要多次工具调用的复杂任务

运行命令：
  python chapter_01_fundamentals/01_hello_agent.py
"""

TEST_CASES = [
    {
        "name": "Test 1 - 简单问答（无需工具）",
        "message": "什么是 Python 编程语言？请用一句话回答。",
        "expected_tools": 0,
    },
    {
        "name": "Test 2 - 天气查询（单工具）",
        "message": "上海今天天气怎么样？",
        "expected_tools": 1,
    },
    {
        "name": "Test 3 - 组合查询（搜索 + 计算）",
        "message": "搜索一下什么是 LangChain，然后帮我算 123 * 456 等于多少。",
        "expected_tools": 2,
    },
    {
        "name": "Test 4 - 需要推理的复杂查询",
        "message": "北京和深圳今天哪个城市更热？温度差多少？",
        "expected_tools": 2,
    },
]


def main():
    """运行所有测试用例。"""
    print("╔══════════════════════════════════════════════════════╗")
    print("║          第1章：第一个 Agent - Hello World           ║")
    print("║          理解 Agent 循环的底层原理                    ║")
    print("╚══════════════════════════════════════════════════════╝")

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n{'#'*60}")
        print(f"# {test['name']}")
        print(f"# 预期工具调用数: {test['expected_tools']}")
        print(f"{'#'*60}")

        try:
            run_agent(test["message"], max_iterations=5)
        except Exception as e:
            print(f"\n❌ 运行出错: {e}")
            print("请检查 .env 文件中的 API Key 配置是否正确。")

        if i < len(TEST_CASES):
            print("\n" + "-" * 60)
            input("按 Enter 继续下一个测试...")


if __name__ == "__main__":
    main()
