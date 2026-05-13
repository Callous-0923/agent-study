"""
第5章：多智能体系统（Multi-Agent System）
=========================================

📌 本章目标：
  1. 理解多智能体系统的核心架构模式
  2. 掌握 Agent 之间的协作/通信机制
  3. 学会用 LangGraph 构建多 Agent 工作流
  4. 了解 crewAI 等框架的多 Agent 编排方式

📌 面试高频点：
  - 多 Agent 系统的架构模式（协作/分层/竞争）
  - Agent 间通信协议设计
  - 如何处理多 Agent 的冲突和一致性问题
  - Multi-Agent 和单 Agent 的取舍


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5.1 为什么需要 Multi-Agent？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**一个类比**：

  单 Agent = 一个全栈工程师
    - 什么都能做，但每项都不够专业
    - 上下文太长，容易分心

  多 Agent = 一个专业团队
    - 每人各司其职（数据分析师 + 程序员 + 文案）
    - 通过分工协作完成复杂任务
    - 每个 Agent 的 prompt 可以高度专业化

核心优势：
  1. 分工明确：每个 Agent 专注自己的领域
  2. 并行执行：多个 Agent 可以同时工作
  3. 错误隔离：一个 Agent 出错不影响整体
  4. 专业化 prompt：每个 Agent 可以用领域最优提示词

经典场景：
  - 软件开发生命周期（需求 → 设计 → 编码 → 测试 → 部署）
  - 内容生产流水线（调研 → 写作 → 编辑 → 发布）
  - 投资分析（宏观分析 → 行业分析 → 个股分析 → 风险评估）

三种经典架构：

┌──────────────┬───────────────────────────────────────┐
│    架构       │               核心思想                  │
├──────────────┼───────────────────────────────────────┤
│ 协作式        │ 多个 Agent 平等对话，共同决策            │
│ (Horizontal) │ 典型：AutoGen 的 GroupChat              │
├──────────────┼───────────────────────────────────────┤
│ 分层式        │ 上级 Agent 委派任务给下级 Agent          │
│ (Vertical)   │ 典型：crewAI 的 Crew + Task 模型        │
├──────────────┼───────────────────────────────────────┤
│ 竞争式        │ 多个 Agent 各自生成方案，择优选用         │
│ (Competitive)│ 典型：辩论式 Agent / Self-Play          │
└──────────────┴───────────────────────────────────────┘
"""

import os
import json
from typing import Annotated, TypedDict, Literal, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5.2 用 LangGraph 构建双 Agent 协作系统
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

场景：内容创作流水线
  - Writer Agent: 负责创作内容
  - Reviewer Agent: 负责审核和改进

流程：
  输入需求 → Writer 创作 → Reviewer 审核 → 
  如果不合格 → Writer 修改 → Reviewer 再审 →
  如果合格 → 输出终稿
"""


class ContentTeamState(TypedDict):
    """内容创作团队的状态。

    所有 Agent 节点共享此状态，通过读写状态来协作。
    这是 LangGraph 多 Agent 系统的核心：
    状态是 Agent 之间的「共享白板」。
    """
    messages: Annotated[list, add_messages]
    task: str                      # 原始任务
    draft: str                     # Writer 的草稿
    review: str                    # Reviewer 的审核意见
    iteration: int                 # 当前迭代轮次
    approved: bool                 # 是否通过审核
    max_iterations: int            # 最大迭代次数


WRITER_SYSTEM_PROMPT = """你是一位专业的内容创作者。
风格要求：
- 语言简洁有力
- 逻辑清晰，论据充分
- 面向技术读者的科普风格
- 字数在 300-500 字之间"""

REVIEWER_SYSTEM_PROMPT = """你是一位严格的内容审核编辑。
审查标准：
- 准确性：内容是否有事实性错误？
- 逻辑性：论述是否严密？
- 可读性：是否通俗易懂？
- 完整性：是否覆盖了任务要求的所有方面？

返回格式：
{"score": 1-10, "issues": ["问题1", "问题2"], 
 "approved": true/false, "suggestions": "改进建议"}"""


def build_content_team(llm):
    """构建内容创作双 Agent 团队。

    Args:
        llm: LangChain ChatOpenAI 实例。

    Returns:
        编译后的 LangGraph 应用。
    """

    def writer_node(state: ContentTeamState) -> ContentTeamState:
        """Writer Agent：创作内容或根据审核意见修改。"""
        iteration = state.get("iteration", 0)
        review = state.get("review", "")

        if review and iteration > 0:
            prompt = (
                f"原始任务: {state['task']}\n\n"
                f"上次草稿: {state['draft']}\n\n"
                f"审核意见: {review}\n\n"
                f"请根据审核意见修改草稿。"
            )
            print(f"  ✏️ [Writer] 第{iteration}轮修改...")
        else:
            prompt = f"任务: {state['task']}\n请撰写内容。"
            print(f"  ✏️ [Writer] 开始创作...")

        response = llm.invoke([
            ("system", WRITER_SYSTEM_PROMPT),
            ("user", prompt),
        ])
        state["draft"] = response.content
        return state

    def reviewer_node(state: ContentTeamState) -> ContentTeamState:
        """Reviewer Agent：审核草稿质量。"""
        print(f"  🔍 [Reviewer] 正在审核...")

        prompt = (
            f"任务要求: {state['task']}\n\n"
            f"待审核草稿:\n{state['draft']}\n\n"
            f"请审核并评分。返回 JSON 格式。"
        )
        response = llm.invoke([
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("user", prompt),
        ])

        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {"score": 5, "issues": ["无法解析审核结果"],
                       "approved": True, "suggestions": ""}

        state["review"] = result.get("suggestions", "")
        state["approved"] = result.get("approved", False)
        print(f"    评分: {result.get('score', 'N/A')}/10 | "
              f"通过: {state['approved']} | "
              f"问题: {result.get('issues', [])}")
        return state

    def should_continue(state: ContentTeamState) -> Literal["writer", "end"]:
        """决定是否继续迭代。

        Returns:
            'writer': 需要 Writer 修改
            'end': 审核通过或达到最大迭代次数
        """
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 3)
        approved = state.get("approved", False)

        if approved:
            print(f"  ✅ 审核通过！总迭代 {iteration + 1} 轮")
            return "end"

        if iteration >= max_iter - 1:
            print(f"  ⚠️ 达到最大迭代次数 {max_iter}，强制结束")
            return "end"

        state["iteration"] = iteration + 1
        return "writer"

    # 构建图
    workflow = StateGraph(ContentTeamState)

    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {"writer": "writer", "end": END},
    )

    return workflow.compile()


"""
图结构示意：

     [START]
        │
        ▼
   ┌─────────┐
   │  Writer  │ ←──────────┐
   └────┬────┘             │
        │ (草稿)            │
        ▼                   │
   ┌──────────┐   不合格     │
   │ Reviewer  │────────────┘
   └────┬─────┘
        │ (合格)
        ▼
     [END]

这是协作式 + 分层式的混合架构：
  - Writer 和 Reviewer 是协作关系（双方一起产出内容）
  - Reviewer 对 Writer 有审查-修改的上下级关系
"""


def demo_content_team():
    """演示双 Agent 内容创作团队。"""
    if not LANGGRAPH_AVAILABLE:
        print("需要安装 langgraph 才能运行此演示。")
        return

    print("\n" + "=" * 60)
    print("  Multi-Agent: 内容创作团队演示")
    print("=" * 60)

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.7,
    )

    app = build_content_team(llm)

    task = "写一篇短文介绍 AI Agent 技术对职场的影响"

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "draft": "",
        "review": "",
        "iteration": 0,
        "approved": False,
        "max_iterations": 3,
    }

    print(f"\n📋 任务: {task}\n")

    try:
        final_state = app.invoke(initial_state)
        print(f"\n{'='*60}")
        print(f"📄 终稿:")
        print(f"{'='*60}")
        print(final_state["draft"])
    except Exception as e:
        print(f"  ⚠️ 运行出错: {e}")


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5.3 crewAI 风格的角色-Crew 模型
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

crewAI 采用「Crew = Agents + Tasks」模型：
  - 定义一系列 Agent（角色）
  - 定义一系列 Task（任务）
  - 将任务分配给 Agent 组成 Crew
  - Crew 自动协调执行

虽然我们不直接安装 crewAI（避免依赖过重），
但可以用 LangChain 模拟其核心概念来理解其设计思想。
"""


class RoleBasedAgent:
    """模拟 crewAI 风格的角色 Agent。

    crewAI 的核心设计理念：
      1. 每个 Agent 有明确的角色(role)、目标(goal)、背景(backstory)
      2. Agent 之间通过任务(task)来协作
      3. 可以设置 allow_delegation 允许 Agent 把任务委派给其他 Agent
    """

    def __init__(self, name: str, role: str, goal: str, backstory: str,
                 llm=None):
        """初始化角色 Agent。

        Args:
            name: Agent 名称。
            role: 角色描述（如 "数据分析师"）。
            goal: 目标描述。
            backstory: 背景故事（帮助 LLM 理解角色定位）。
            llm: LangChain LLM 实例。
        """
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm

    def execute(self, task: str) -> str:
        """执行分配给此 Agent 的任务。

        Args:
            task: 任务描述。

        Returns:
            执行结果。
        """
        system_prompt = (
            f"你是 {self.name}，一名 {self.role}。\n"
            f"你的目标: {self.goal}\n"
            f"你的背景: {self.backstory}"
        )
        if self.llm is None:
            return f"[{self.name}] 模拟输出：已完成任务 '{task}'"
        response = self.llm.invoke([
            ("system", system_prompt),
            ("user", task),
        ])
        return response.content


def demo_crew_style():
    """演示 crewAI 风格的多 Agent 协作。

    场景：市场分析报告
      - 数据分析师：收集和处理数据
      - 策略规划师：基于数据制定策略
      - 报告撰写师：汇总形成最终报告
    """
    print("\n" + "=" * 60)
    print("  crewAI 风格: 市场分析团队演示")
    print("=" * 60)

    # 尝试初始化 LLM
    try:
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.5,
        )
    except Exception:
        llm = None
        print("  ⚠️ 未配置 API Key，使用模拟模式")

    # 定义团队
    data_analyst = RoleBasedAgent(
        name="数据分析师",
        role="市场数据分析专家",
        goal="收集并分析市场数据，找出关键趋势和洞察",
        backstory="具有10年市场研究经验，擅长从数据中发现规律",
        llm=llm,
    )

    strategist = RoleBasedAgent(
        name="策略规划师",
        role="商业策略专家",
        goal="基于数据洞察制定可行的商业策略",
        backstory="曾任多家顶级咨询公司顾问，擅长战略规划",
        llm=llm,
    )

    report_writer = RoleBasedAgent(
        name="报告撰写师",
        role="专业报告撰写人",
        goal="将分析和策略整合为一份清晰、专业的研究报告",
        backstory="资深商业撰稿人，擅长将复杂信息转化为易读报告",
        llm=llm,
    )

    # 流程：数据分析 → 策略制定 → 报告撰写
    print("\n📊 阶段 1: 数据分析")
    analysis = data_analyst.execute(
        "分析2025年中国AI Agent市场的规模和主要趋势，列出3-5个关键发现"
    )
    print(f"    输出: {analysis[:150]}...")

    print("\n📈 阶段 2: 策略制定")
    strategy = strategist.execute(
        f"基于以下市场分析，制定一套市场进入策略:\n{analysis}"
    )
    print(f"    输出: {strategy[:150]}...")

    print("\n📝 阶段 3: 报告撰写")
    report = report_writer.execute(
        f"请将以下分析报告和策略建议整合为一份正式研究报告:\n\n"
        f"市场分析:\n{analysis}\n\n策略建议:\n{strategy}\n\n"
        f"要求: 结构清晰、有执行摘要、分章节、每个观点有数据支撑。"
    )
    print(f"    输出: {report[:200]}...")

    print("\n📄 ===== 最终报告摘要 =====")
    print(report if len(report) < 500 else report[:500] + "...")


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5.4 Multi-Agent 设计原则（面试高频！）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 职责单一原则
   - 每个 Agent 只做一件事，做到极致
   - 不要给一个 Agent 塞太多职责

2. 通信标准化
   - Agent 之间用结构化格式通信（JSON/XML）
   - 避免自然语言的模糊性

3. 错误隔离
   - 一个 Agent 失败不应导致整个系统崩溃
   - 设置重试机制和降级策略

4. 可观测性
   - 记录每个 Agent 的输入/输出/耗时
   - 便于调试和优化

5. 成本控制
   - Multi-Agent 会成倍增加 API 调用量
   - 用缓存、batching、模型分层来降低成本

面试常问：
  "Multi-Agent 比单 Agent 好在哪里？什么时候不该用？"

  好处：
    - 分工专业化，输出质量更高
    - 复杂任务可以并行处理
    - 系统更模块化、易维护

  不适用场景：
    - 简单任务（杀鸡用牛刀）
    - 成本敏感场景（多倍 API 调用）
    - Agent 间协调开销 > 收益的场景
"""


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5.5 本章总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心要点回顾：

1. Multi-Agent = 多角色协作
   - 协作式：平等对话，共同决策
   - 分层式：上级委派，下级执行
   - 竞争式：多方生成，择优选用

2. 通信机制
   - 共享状态（LangGraph 的方式）
   - 消息传递（AutoGen 的方式）
   - 任务委派（crewAI 的方式）

3. 实践框架
   - LangGraph：灵活构建自定义 Multi-Agent 图
   - crewAI：开箱即用的角色-任务模型
   - AutoGen：对话驱动的 Multi-Agent

4. 设计原则
   - 职责单一、通信标准化、错误隔离、可观测、控成本

面试速记：
  "你们项目用了 Multi-Agent 吗？为什么？"
  → 描述场景复杂度（需要多角色协作）
  → 说明架构设计（用什么模式、为什么）
  → 展示效果提升（效率/质量提升多少）
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║       第5章：多智能体系统 (Multi-Agent)               ║")
    print("║       双Agent协作 · crewAI风格 · 设计原则            ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n▶ 5.2 双 Agent 协作（Writer + Reviewer）")
    try:
        demo_content_team()
    except Exception as e:
        print(f"  ⚠️ 运行出错（需要 API Key）: {e}")

    print("\n▶ 5.3 crewAI 风格角色 Agent")
    try:
        demo_crew_style()
    except Exception as e:
        print(f"  ⚠️ 运行出错: {e}")

    print("\n▶ 5.4 Multi-Agent 速查表")
    print("-" * 50)
    print("架构: 协作式 → 平等 Agent 共同决策")
    print("架构: 分层式 → 上级委派任务给下级")
    print("架构: 竞争式 → 多方生成，择优选用")
    print("框架: LangGraph → 自定义 Multi-Agent 图")
    print("框架: crewAI   → 角色-任务模型")
    print("框架: AutoGen  → 对话驱动 Multi-Agent")
    print("原则: 职责单一、通信标准、错误隔离、可观测、控成本")

    print("\n✅ 第5章完成！接下来进入第6章：评估与最佳实践。")
