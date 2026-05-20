"""
第11章：Tool Calling 底层机制深度剖析
=====================================

📌 本章目标：
  1. 深入理解 LLM Function Calling 的「真正原理」
  2. 掌握 OpenAI / Anthropic 两套实现的技术细节和差异
  3. 理解 Streaming Tool Calls 的机制
  4. 学会 Parallel Tool Calling 的使用与限制
  5. 掌握 Strict Function Calling（严格模式）
  6. 理解 Tool Calling 中的常见陷阱和解决方案

📌 面试高频点：
  - LLM 是如何「知道」该调用哪个工具的？
  - OpenAI 的 function calling 和 Anthropic 的 tool_use 有什么区别？
  - 什么是 Parallel Tool Calling？什么时候该用，什么时候不该用？
  - Streaming 模式下如何处理 Tool Call？
  - Token 消耗：tool definitions 计入 prompt token 吗？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本章是基于 OpenAI / Anthropic API 文档 + 实践经验的总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


11.1 Function Calling 真的不是 LLM 在「调用函数」！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是最常见的误解。让我们从底层理解真正发生了什么。

误解：「LLM 调用了我的函数」
真相：「LLM 输出了一个 JSON 对象，你的代码解析了它并执行了函数」

类比：
  你给朋友发短信：「如果需要打车，回复格式 {action: "call_taxi", from: "...", to: "..."}」
  朋友回复：{"action": "call_taxi", "from": "天安门", "to": "机场"}

  你做的事：
    1. 解析这条 JSON
    2. 调用打车 API("天安门", "机场")
    3. 告诉朋友结果

  LLM 做的事：
    1. 根据你的 tool definition，决定是否使用工具
    2. 返回一个结构化的 JSON（而不是自然语言）
    3. 你的代码解析 JSON 并执行真正的函数

所以 Function Calling 本质上是：
  「LLM 的结构化输出能力」+「协议规约」
  不是 LLM 有了「执行函数」的能力！


11.2 Token 层面的原理 —— 工具定义去哪了？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当你在 API 请求中包含 tools 参数时：

  POST /v1/chat/completions
  {
    "model": "gpt-4o",
    "messages": [...],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "查询天气",
          "parameters": {...}
        }
      }
    ]
  }

OpenAI 的处理流程：
  1. 将 tools 定义序列化为文本
  2. 将序列化后的文本拼接到 system prompt 后面
  3. 拼接内容会计入 prompt token（扣费！）

Anthropic 的处理流程类似但有差异：
  1. 将 tools 作为独立字段发送
  2. 模型在训练时已学会理解 tool schema
  3. 同样计入 prompt token

面试重点：tools 定义消耗 token！
  - 一个典型工具的 JSON Schema 约 200-500 tokens
  - 10 个工具可能消耗 2000-5000 tokens
  - 这是每次请求的固定开销！
  - 优化策略：只传当前场景可能用到的工具（动态工具选择）


11.3 OpenAI vs Anthropic —— 两套实现对比
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────────┬───────────────────────┬──────────────────────────┐
│      维度         │  OpenAI Function Call  │  Anthropic Tool Use      │
├──────────────────┼───────────────────────┼──────────────────────────┤
│ 工具定义格式      │ 嵌套对象               │ 更简洁的 Tool 对象        │
│                  │ {type:"function",     │ {name:"...",            │
│                  │  function:{...}}      │  description:"...",      │
│                  │                       │  input_schema:{...}}     │
├──────────────────┼───────────────────────┼──────────────────────────┤
│ 返回格式          │ tool_calls 数组        │ content 中的 tool_use    │
│                  │ [{"function":{...}}]  │ blocks                  │
├──────────────────┼───────────────────────┼──────────────────────────┤
│ 并行调用          │ 同时返回多个 tool_call  │ 同时返回多个 tool_use    │
│                  │ (parallel_tool_calls  │ block                   │
│                  │  参数控制)             │                         │
├──────────────────┼───────────────────────┼──────────────────────────┤
│ 流式支持          │ 增量式 tool_call 名称   │ content_block 流式输出   │
│                  │ 和参数                  │                         │
├──────────────────┼───────────────────────┼──────────────────────────┤
│ 严格模式          │ strict: true          │ 不支持单独的 strict 模式   │
│                  │ (保证 JSON Schema 一致) │ (但通过 prompt 可实现)    │
├──────────────────┼───────────────────────┼──────────────────────────┤
│ Token 计算        │ tools 序列化拼入 prompt │ tools 作为独立参数       │
├──────────────────┼───────────────────────┼──────────────────────────┤
│ 工具结果返回      │ Tool Message           │ Tool Result Content Block│
│                  │ role="tool"            │ type="tool_result"      │
└──────────────────┴───────────────────────┴──────────────────────────┘

两种格式对比（JSON）：

  # OpenAI 格式
  {
    "role": "assistant",
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"北京\"}"
        }
      }
    ]
  }

  # Anthropic 格式
  {
    "role": "assistant",
    "content": [
      {
        "type": "tool_use",
        "id": "toolu_abc123",
        "name": "get_weather",
        "input": {"city": "北京"}
      }
    ]
  }

关键差异：
  - OpenAI: arguments 是 JSON 字符串（需要 json.loads）
  - Anthropic: input 直接是 JSON 对象
"""

# 两个平台的工具定义格式对比
OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"],
        },
    },
}

ANTHROPIC_TOOL = {
    "name": "get_weather",
    "description": "查询指定城市的天气信息。返回温度、湿度、天气状况。",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，如 '北京'、'上海'",
            }
        },
        "required": ["city"],
    },
}


"""
11.4 Parallel Tool Calling —— 并行执行
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当 LLM 判断多个工具调用之间没有依赖关系时，
它可以一次性返回多个 tool_call，由你的代码并行执行。

示例场景：
  用户：「北京和上海的天气分别怎么样？」

  LLM 返回 2 个 tool_call（同一个 response）:
    tool_call_1: get_weather("北京")
    tool_call_2: get_weather("上海")

  你的代码并行执行这两个请求（节省时间！）

但需要注意：
  1. 并行执行的前提：工具之间没有依赖关系
  2. 如果 tool_2 依赖 tool_1 的结果 → 必须串行
  3. LLM 自己会判断（通过参数推断依赖关系）

控制行为：
  OpenAI: tool_choice 参数
    - "auto": LLM 自行决定（默认）
    - "required": 必须至少调用一个工具
    - "none": 不允许调用工具
    - {"type": "function", "function": {"name": "get_weather"}}: 强制调用指定工具

  Anthropic: tool_choice 参数
    - "auto": LLM 自行决定（默认）
    - "any": 必须至少调用一个工具
    - "tool": 强制调用指定工具
    - 不支持 "none"（没有等效设置）


11.5 Streaming Tool Calls —— 流式工具调用
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在 streaming 模式下，tool_call 是「增量式」到达的：

  OpenAI streaming 过程：
    chunk_1: {"delta": {"tool_calls": [{"index": 0, "id": "call_abc"}]}}
    chunk_2: {"delta": {"tool_calls": [{"index": 0, "function": {"name": "get"}}]}}
    chunk_3: {"delta": {"tool_calls": [{"index": 0, "function": {"name": "_weather"}}]}}
    chunk_4: {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"cit"}}]}}
    chunk_5: {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "y\":\"北京\"}} ]}}
    ...

  我们需要「组装」这些增量片段为完整的 tool_call。

  Anthropic streaming 过程：
    chunk_1: {"type": "content_block_start", "content_block": {"type": "tool_use", ...}}
    chunk_2: {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{\"city\""}}
    chunk_3: {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": ":\"北京\"}"}}
    ...

关键挑战：
  1. 需要维护状态机（哪些 tool_call 是同一个？）
  2. 参数是增量到达的，需要累积拼接
  3. 可能同时收到多个 tool_call 的流（用 index 区分）


11.5.1 Streaming 组装实战 —— 面试官想知道你处理过脏数据
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▍ 状态机设计 —— 不只是「拼接字符串」

  面试常问：「Streaming 模式下怎么组装 tool_call？」

  简单回答是「用 index 区分 + 累积 arguments 字符串」。但真实项目中
  需要处理至少 5 种异常状态：

  异常 1：丢包乱序 —— 网络抖动导致 chunk_5 先到、chunk_3 后到
    对策：按 index + sequence_number（如果有）排序重组
    
  异常 2：中断恢复 —— 流中断后重新连接，tool_call 拼接了一半
    对策：保存中间状态，重连后从断点继续
    
  异常 3：JSON 解析失败 —— arguments 拼接完成但是无效 JSON
    对策：regex 修复常见错误（尾逗号、单引号）→ 重试 → 放弃
    
  异常 4：并行 tool_call 的流交错 —— 3 个 tool_call 同时流式到达
    对策：用 index 分桶，各自独立组装，互不干扰
    
  异常 5：空参数 —— LLM 某些情况下返回没有 arguments 的 tool_call
    对策：加默认空对象 {}，不 crash

▍ Anthropic 的 content_block 设计为什么更优雅？

  Anthropic 用 content_block_start/delta/stop 三阶段模型：
    - start → 声明「一个 tool_use block 开始了」，type + id 已知
    - delta → 流式传输 input_json_delta（只传增量 JSON）
    - stop → 声明这个 block 结束
    
  比 OpenAI 的 index 模式好在哪里？
    - 不需要 index 来区分并行调用（不同的 content_block 就是不同的）
    - 生命周期明确 → 你在 stop 事件时可以确认「完整了」
    - 可以处理非 JSON 的 block 类型（如文本 block + 工具 block 交替）

  面试可以提：「Anthropic 的 content_block 流式模型给每个 tool_call
  分配了明确的生命周期，避免了 OpenAI index 模式下的并行组装复杂度。」

▍ 生产中的 Token 陷阱 —— tools 定义在「吃」你的预算

  很多人不知道 tools 定义每次请求都会重复计费。一个 Agent 如果有
  15 个工具，每个工具 300 tokens 的定义 → 4500 tokens/次的前缀开销。
  
  优化策略：
    a) 动态工具集 —— 第一轮不带 tools，让 LLM 说「我需要什么」
       → 第二轮只带 2-3 个相关工具
    b) 工具描述瘦身 —— description 越短越好（Anthropic 建议 < 200 字）
    c) Schema 惰性加载 —— 只在 LLM 选择了某个工具后才传完整 Schema
    d) 缓存工具列表 —— 使用 Prompt Caching（Ch33）缓存不变的 tools 前缀


11.6 Strict Function Calling —— 严格模式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OpenAI 在 2024 年推出了 strict 模式：
  设置 strict: true 时，LLM 保证输出的参数 100% 符合 JSON Schema。

▍ Strict 模式的代价 —— 面试官想听你说「限制」

  面试常见陷阱：「你用 strict 模式了吗？」→ 如果你只说「用了，保证格式」，
  就太浅了。正确的深度回答应当包含 strict 模式的限制：

  1. 所有参数必须定义在顶层 object —— 不支持嵌套的 anyOf/oneOf
  2. 所有字段必须显式声明 type + 是否 nullable
  3. 不支持动态 Schema —— 「根据上一个参数决定这个参数的类型」
     这种场景 Strict 做不了
  4. 可选字段必须声明 default 或 nullable，否则 LLM 可能不生成
  5. 额外限制：顶层必须为 object 类型，数组类型参数需包装在 object 内

  面试可以提：「Strict 模式适合 API 网关类的确定性工具，
  但对需要 LLM 灵活输出结构（如图表配置、动态查询）的场景
  反而会限制表达能力。我通常按工具类型分：确定性工具用 strict，
  创造性工具不用。」

▍ 为什么 Anthropic 没有 Strict 模式？
  
  因为 Anthropic 的设计哲学不同：不是靠 Schema 约束，而是
  通过训练让模型「学会」输出正确格式的 JSON。Claude 3.5+
  在工具调用格式正确率上已经非常接近 100%（非 strict 条件下）。
  
  实践中：Claude 的 tool_use 返回的 input 字段已经是解析好的
  JSON 对象（不是字符串），这本身减少了一半的格式错误。


11.6.1 Tool Calling 生产 Checklist
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  你上线前应该检查：
  ✅ 每个工具的 description < 200 字（Anthropic 建议）
  ✅ 动态加载工具，不是一次传 20 个
  ✅ 参数有明确的 type + default + nullable 声明
  ✅ 工具返回有 max_length 限制（避免 10MB JSON 爆上下文）
  ✅ 工具超时设置（>10s 的工具调异步模式）
  ✅ 幂等性保证（相同参数多次调用 = 相同结果或安全重试）
  ✅ 错误返回统一格式（{"error": "...", "retryable": true/false}）

如何使用：
  - 所有参数必须定义为 object 类型
  - 所有字段必须有明确的 type
  - 可选字段必须设置 default 或 nullable
  - 不允许使用 anyOf / oneOf

示例：
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "strict": true,  // ← 开启严格模式
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city", "unit"],
        "additionalProperties": false  // ← strict 模式必须设置
      }
    }
  }

为什么需要 strict 模式？
  1. 生产环境要求：参数格式错误会导致工具执行失败
  2. 减少重试：不需要「尝试解析 → 失败 → 重新请求 LLM」
  3. 安全考虑：防止 LLM 输出未定义的字段

Anthropic 目前没有单独的 strict 参数，
但可以通过 prompt 工程实现类似效果：
  "You MUST output exactly the parameters specified in the input_schema.
   Do not include any additional fields."


11.7 Tool Calling 的进阶技巧
━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 工具描述即 Prompt 工程
   - description 不只是给开发者看的，是给 LLM 看的！
   - 描述要回答三个问题：
     a. 这个工具「做什么」？
     b. 「什么时候」该用？
     c. 「什么时候」不该用？

2. 参数设计原则
   - 用 enum 限制选项（减少幻觉）
   - 提供清晰的参数描述
   - 必填 vs 可选要明确
   - 给出典型调用示例（在 description 中）

3. 错误处理 —— 工具结果应「帮助 LLM 纠正」
   ✗ return "Error 404"
   ✓ return "未找到城市'培京'的天气数据。城市名是否拼写有误？可用城市：北京、上海。"

4. 工具数量控制 —— 别给 LLM 太多选择
   - 超过 10 个工具时，LLM 的选择准确率显著下降
   - 采用「动态工具集」：根据当前上下文过滤可用工具
   - 分层的工具注册：先给概览，用户指定后再给详细工具

5. 工具调用确认（Human-in-the-Loop）
   - 读操作：自动执行（get_weather, search）
   - 写操作：需要确认（send_email, delete_file）
   - 危险操作：需要二次确认（execute_sql, run_command）
"""


class ToolCallSimulator:
    """模拟两个平台的 Tool Call 处理流程。

    展示从 LLM 返回 → 解析 → 执行 → 返回结果的完整链路。
    """

    def __init__(self):
        self.tools = {
            "get_weather": lambda args: f"天气: {args.get('city', '未知')} 晴 25°C",
            "search": lambda args: f"搜索结果(关于'{args.get('query', '')}'): ...",
            "calculate": lambda args: str(eval(args.get('expression', '0'))),
        }

    def process_openai_style(self, llm_response: dict) -> str:
        """处理 OpenAI 风格的 tool_calls。

        Args:
            llm_response: 模拟的 OpenAI LLM 响应。

        Returns:
            执行结果描述。
        """
        tool_calls = llm_response.get("tool_calls", [])
        if not tool_calls:
            return "LLM 直接回答: " + llm_response.get("content", "")

        results = []
        can_parallel = True

        # 检查是否可以并行（简化：都假设可并行）
        for tc in tool_calls:
            func = tc["function"]
            tool_name = func["name"]
            # OpenAI 的 arguments 是 JSON 字符串！
            args = __import__("json").loads(func["arguments"])

            print(f"  [OpenAI] 调用: {tool_name}({args})")

            if tool_name in self.tools:
                result = self.tools[tool_name](args)
            else:
                result = f"错误: 未知工具 {tool_name}"

            results.append({
                "tool_call_id": tc["id"],
                "role": "tool",
                "content": result,
            })
            print(f"    结果: {result}")

        # 组装返回给 LLM 的工具结果
        return str(results)

    def process_anthropic_style(self, llm_response: dict) -> str:
        """处理 Anthropic 风格的 tool_use blocks。

        Args:
            llm_response: 模拟的 Anthropic LLM 响应。

        Returns:
            执行结果描述。
        """
        content_blocks = llm_response.get("content", [])
        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

        if not tool_uses:
            text_blocks = [b for b in content_blocks if b.get("type") == "text"]
            return "LLM 直接回答: " + "".join(
                b.get("text", "") for b in text_blocks
            )

        results = []
        for tu in tool_uses:
            tool_name = tu["name"]
            # Anthropic 的 input 直接是 dict（不需要 json.loads）
            args = tu["input"]

            print(f"  [Anthropic] 调用: {tool_name}({args})")

            if tool_name in self.tools:
                result = self.tools[tool_name](args)
            else:
                result = f"错误: 未知工具 {tool_name}"

            results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result,
            })
            print(f"    结果: {result}")

        return str(results)


def demo_tool_calling_flow():
    """演示完整的 Tool Calling 流程。"""
    print("=" * 60)
    print("  Tool Calling 完整流程演示")
    print("=" * 60)

    simulator = ToolCallSimulator()

    # 模拟 OpenAI 风格
    print("\n  [OpenAI 风格 Tool Call]")
    openai_response = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "北京"}',
                },
            },
            {
                "id": "call_def456",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"expression": "25 + 30"}',
                },
            },
        ],
    }
    simulator.process_openai_style(openai_response)

    # 模拟 Anthropic 风格
    print("\n  [Anthropic 风格 Tool Use]")
    anthropic_response = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_xyz789",
                "name": "search",
                "input": {"query": "AI Agent 最新进展"},
            },
        ],
    }
    simulator.process_anthropic_style(anthropic_response)


def demo_streaming_assembly():
    """模拟 Streaming Tool Call 的组装过程。"""
    print("\n" + "=" * 60)
    print("  Streaming Tool Call 组装演示")
    print("=" * 60)

    # 模拟 OpenAI streaming chunks
    chunks = [
        {"delta": {"tool_calls": [{"index": 0, "id": "call_abc"}]}},
        {"delta": {"tool_calls": [{"index": 0, "function": {"name": "get"}}]}},
        {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"cit"}}]}},
        {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "y\":\"北京\"}"}}]}},
    ]

    print("\n  接收到的 streaming chunks:")
    tool_calls_state = {}
    for i, chunk in enumerate(chunks):
        print(f"  chunk_{i}: {chunk}")
        delta = chunk.get("delta", {})
        tool_deltas = delta.get("tool_calls", [])
        for td in tool_deltas:
            idx = td.get("index", 0)
            if idx not in tool_calls_state:
                tool_calls_state[idx] = {
                    "id": td.get("id", ""),
                    "name": "",
                    "arguments": "",
                }
            if "function" in td:
                if "name" in td["function"]:
                    tool_calls_state[idx]["name"] += td["function"]["name"]
                if "arguments" in td["function"]:
                    tool_calls_state[idx]["arguments"] += td["function"]["arguments"]

    print("\n  组装后的完整 tool_call:")
    for idx, tc in tool_calls_state.items():
        print(f"  tool_call[{idx}]: {tc['name']}({tc['arguments']})")


"""
11.8 本章总结
━━━━━━━━━━━━━━

核心要点回顾：

1. Function Calling 的本质
   - LLM 输出结构化 JSON，不是真的「调用」函数
   - 工具定义会计入 prompt token（成本）
   - 工具描述是 LLM 选择工具的唯一依据（Prompt 工程）

2. OpenAI vs Anthropic
   - arguments: JSON 字符串 vs 直接对象
   - strict mode: OpenAI 有，Anthropic 靠 prompt
   - 底层原理相同，API 形式不同

3. Parallel Tool Calling
   - 无依赖的工具可以并行执行
   - LLM 自己判断依赖关系
   - 用 tool_choice 参数控制行为

4. Streaming Tool Calls
   - 增量式到达，需要组装
   - 用 index 区分多个 tool_call
   - 需要维护状态机

5. 生产建议
   - 严格模式（当可用时）
   - 错误信息帮助 LLM 自我纠正
   - 工具分读写权限，写入需确认
   - 动态工具选择（减少 token 浪费）

面试速记：
  "Function Calling 的原理是什么？"
  → LLM 根据 tools 定义判断是否需要调用工具
  → 输出结构化 JSON（包含工具名和参数）
  → 开发者的代码执行真正的函数
  → 将执行结果返回给 LLM
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第11章：Tool Calling 底层机制深度剖析                 ║")
    print("║  OpenAI/Anthropic · Parallel · Streaming · Strict   ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n▶ 11.3 两个平台的工具定义格式对比")
    print("-" * 50)
    import json
    print("OpenAI:")
    print(json.dumps(OPENAI_TOOL, indent=2, ensure_ascii=False)[:200] + "...")
    print("\nAnthropic:")
    print(json.dumps(ANTHROPIC_TOOL, indent=2, ensure_ascii=False)[:200] + "...")

    print("\n▶ 11.4-11.6 Tool Calling 流程演示")
    demo_tool_calling_flow()

    print("\n▶ 11.5 Streaming 组装演示")
    demo_streaming_assembly()

    print("\n▶ 11.7 进阶技巧总结")
    tips = [
        "工具描述 = Prompt 工程（告诉 LLM 何时用、何时不用）",
        "用 enum 限制参数 = 减少幻觉",
        "错误信息要能帮助 LLM 自我纠正",
        "工具超过10个 → 动态过滤",
        "写入操作 → 人工确认",
        "strict 模式 → 生产环境必用",
    ]
    for t in tips:
        print(f"  🔑 {t}")

    print("\n✅ 第11章完成！")
