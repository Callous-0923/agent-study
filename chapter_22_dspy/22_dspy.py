"""
第22章：DSPy —— Prompt 不是手写的，是编译出来的
==================================================

📌 本章目标：
  1. 理解 DSPy 的核心哲学：「Programming, not Prompting」
  2. 掌握 Signature → Module → Optimizer 三层抽象
  3. 学会用 DSPy 的编译思想优化 Agent 的 Prompt
  4. 理解 DSPy 和 LangChain 的关系和互补

📌 面试高频点：
  - 「DSPy 是什么？解决了什么问题？」
  - 「Signature 和传统 Prompt 有什么区别？」
  - 「Optimizer 是怎么工作的？」
  - 「DSPy 和 LangChain 怎么配合？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DSPy: Stanford NLP 实验室，github.com/stanfordnlp/dspy
核心理念来自论文: "DSPy: Compiling Declarative LM Calls into Self-Improving Pipelines"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


22.1 编程 vs 提示词 —— DSPy 的设计哲学
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统 Prompt 工程的问题：

  你写了一个 Agent：
    prompt = "你是一个助手，请用中文回答..."

  换一个模型（GPT-4 → Claude）：
    → Prompt 要重新调整
    → 效果可能大幅下降
    → 这就是「Prompt 脆弱性」

DSPy 的解决方案：
  把 Prompt 当作「代码的编译产物」而非手写字符串

  类比：
    传统 Prompt = 手动写汇编代码（每换一个 CPU 就要重写）
    DSPy = 写 C 代码再编译（代码不变，编译器适配不同 CPU）

  翻译成 DSPy 术语：
    DSPy 程序（Python 代码）→ Optimizer（编译器）→ Prompt（机器码）
                                           ↑
                                    自动适配不同的 LLM


22.2 Signature —— 把 Prompt 变成「函数签名」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统 Prompt：
  prompt = "给定问题 {question}，请给出答案"  # 脆弱的字符串！

DSPy Signature：
  class QA(dspy.Signature):
      question = dspy.InputField(desc="问题")
      answer = dspy.OutputField(desc="答案")

Signature 的优势：
  1. 声明式：描述「输入什么、输出什么」，不碰字符串
  2. 自文档化：desc 即文档，人类和 Optimizer 都能理解
  3. 可组合：Signature 可以拼接成复杂流水线
  4. 可验证：输出类型自动校验
"""

import json
import time
import hashlib
from typing import Optional
from dataclasses import dataclass, field


# ===== 模拟 DSPy 核心概念（不 import dspy）=====

@dataclass
class SignatureField:
    """Signature 字段 —— 带描述的类型标注。"""
    desc: str = ""


def InputField(desc: str = "") -> SignatureField:
    """标记输入字段。"""
    return SignatureField(desc=desc)


def OutputField(desc: str = "") -> SignatureField:
    """标记输出字段。"""
    return SignatureField(desc=desc)


class DSPySignature:
    """模拟 DSPy 的 Signature 基类。

    真实 DSPy 中：
      class QA(dspy.Signature):
          question = dspy.InputField()
          answer = dspy.OutputField()
    """

    @classmethod
    def describe(cls) -> dict:
        """生成签名描述（给 Optimizer 用）。"""
        fields = {}
        for name, val in cls.__dict__.items():
            if isinstance(val, SignatureField):
                fields[name] = {
                    "role": "input" if isinstance(val, InputField.__class__)
                            else "output",
                    "desc": val.desc,
                }
        return {
            "name": cls.__name__,
            "fields": fields,
        }

    @classmethod
    def to_prompt_template(cls) -> str:
        """将 Signature 编译为 Prompt 模板。

        这是 DSPy Optimizer 的核心能力：
          根据 Signature → 自动生成最优的 Prompt 结构。
        """
        desc = cls.describe()
        inputs = [f"{k}: {v['desc']}" for k, v in desc["fields"].items()
                  if v["role"] == "input"]
        outputs = [f"{k}: {v['desc']}" for k, v in desc["fields"].items()
                   if v["role"] == "output"]

        return (
            f"## 任务: {desc['name']}\n\n"
            f"### 输入\n" + "\n".join(f"- {i}" for i in inputs) +
            f"\n\n### 输出\n" + "\n".join(f"- {o}" for o in outputs) +
            f"\n\n请严格按输出格式返回 JSON。"
        )


"""
22.3 Module —— 把 Agent 组件变成「函数」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DSPy Module 是可组合的 AI 组件。

内置 Module 类型：

  ┌──────────────────┬──────────────────────────────────┐
  │ Module            │        作用                       │
  ├──────────────────┼──────────────────────────────────┤
  │ dspy.Predict      │ 最基本的单步 LLM 调用             │
  │ dspy.ChainOfThought│ 带推理链的 LM 调用               │
  │ dspy.ReAct        │ ReAct Agent 循环                 │
  │ dspy.MultiChain   │ 多链并行 + 投票                  │
  │ dspy.ProgramOfThought│ 用代码辅助推理                │
  └──────────────────┴──────────────────────────────────┘

类比 LangChain：
  LangChain Chain ≈ DSPy Module
  区别：DSPy Module 的参数（prompts/weights）可由 Optimizer 自动调优

Agent 开发者用 DSPy 的样子：

  class AgentQA(dspy.Module):
      def __init__(self):
          self.search = dspy.ChainOfThought("query -> results")
          self.answer = dspy.ChainOfThought("context, question -> answer")

      def forward(self, question):
          results = self.search(query=question)
          return self.answer(context=results, question=question)
"""


class DSPyModule:
    """模拟 DSPy Module —— 可组合的 AI 组件。"""

    def __init__(self, name: str, signature: type):
        self.name = name
        self.signature = signature
        self.prompt = signature.to_prompt_template()
        self.optimized_prompt = self.prompt  # Optimizer 会修改这个值
        self.few_shot_examples = []  # Optimizer 会填充这个列表

    def __call__(self, **inputs) -> dict:
        """执行模块（模拟 LLM 调用）。"""
        # 组装最终 prompt
        examples_text = ""
        if self.few_shot_examples:
            examples_text = "\n\n### 示例\n" + "\n---\n".join(
                json.dumps(ex, ensure_ascii=False)
                for ex in self.few_shot_examples[-3:]
            )

        final_prompt = self.optimized_prompt + examples_text

        # 模拟 LLM 返回
        return self._mock_llm(final_prompt, inputs)

    def _mock_llm(self, prompt: str, inputs: dict) -> dict:
        """模拟 LLM 响应（真实实现调用 OpenAI/Claude）。"""
        output_fields = [k for k, v in self.signature.describe()["fields"].items()
                         if v["role"] == "output"]
        result = {}
        for field in output_fields:
            result[field] = f"[模拟] 基于 {len(prompt)} 字符模板生成"
        return result


"""
22.4 Optimizer (Teleprompter) —— 自动优化 Prompt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是 DSPy 最革命性的部分。

传统做法（手调 Prompt）：
  1. 写一个 Prompt
  2. 在 5 个例子上测试
  3. 效果不好 → 修改 Prompt → 再测试
  4. 换模型 → 全部重来

DSPy 做法（自动优化）：
  1. 定义 Metric（评分函数）
  2. 准备训练集（10 个标注样本就够）
  3. 运行 Optimizer → 自动调出最优 Prompt

DSPy 的 Optimizer 类型：

  ┌─────────────────────┬──────────────────────────────────┐
  │ Optimizer            │         做什么                    │
  ├─────────────────────┼──────────────────────────────────┤
  │ BootstrapFewShot    │ 自动找出最佳少样本示例              │
  │ MIPROv2             │ 自动探索最优指令 + 示例组合         │
  │ BootstrapFinetune   │ 为每个 Module 自动微调小模型        │
  │ BetterTogether      │ 组合多个 Optimizer 级联优化         │
  └─────────────────────┴──────────────────────────────────┘

工作流程（MIPROv2 为例）：

  1. Bootstrapping 阶段：
     运行未优化的程序多次 → 收集成功轨迹
     → 筛选出得分高的输入输出对 → 作为候选示例

  2. Grounded Proposal 阶段：
     分析程序代码 + 数据 + 轨迹
     → 生成多个候选指令（instruction variants）

  3. Discrete Search 阶段：
     在训练集上尝试不同「指令 + 示例」的组合
     → 用 Metric 评分 → 选出最优组合
"""


class DSPyOptimizer:
    """模拟 DSPy Optimizer —— 演示自动优化的工作流。

    三样东西：
      1. 你的 DSPy 程序（Module）
      2. Metric（评分函数）
      3. 训练输入（少量带标注的样本）
    """

    def __init__(self, metric: callable):
        """
        Args:
            metric: 评分函数: (predicted, expected) -> score (0-1)
        """
        self.metric = metric
        self.candidates = []

    def compile(self, module: DSPyModule,
                trainset: list[dict],
                max_candidates: int = 5) -> DSPyModule:
        """编译（优化）一个 Module。

        流程：
          1. 收集轨迹（多次运行，记录输入输出）
          2. 筛选高分轨迹
          3. 生成候选 Prompt 变体
          4. 在训练集上评估，选最优

        Args:
            module: 要优化的 Module。
            trainset: 训练数据（输入+期望输出）。
            max_candidates: 生成的候选数。

        Returns:
            优化后的 Module。
        """
        # 阶段 1：Bootstrapping（收集成功案例）
        trajectories = []
        for sample in trainset:
            inputs = sample.get("inputs", {})
            expected = sample.get("outputs", {})
            result = module(**inputs)
            score = self.metric(result, expected)
            if score > 0.6:
                trajectories.append({
                    "inputs": inputs,
                    "outputs": result,
                    "expected": expected,
                    "score": score,
                })

        # 阶段 2：筛选 best-shot 示例
        trajectories.sort(key=lambda t: t["score"], reverse=True)
        best_trajectories = trajectories[:max_candidates]

        # 阶段 3：注入优化结果
        module.few_shot_examples = [
            {"Q": t["inputs"], "A": t["outputs"]}
            for t in best_trajectories
        ]

        # 阶段 4：生成优化后的指令（简化版）
        if best_trajectories:
            avg_score = sum(t["score"] for t in best_trajectories) / len(best_trajectories)
            module.optimized_prompt = (
                module.signature.to_prompt_template() +
                f"\n\n[优化提示: 基于 {len(best_trajectories)} 个样本调优, "
                f"平均得分 {avg_score:.1%}]"
            )

        self.candidates = best_trajectories
        return module


"""
22.5 DSPy + Agent 的实战价值
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DSPy 不是 LangChain 的替代品，是补充：

  DSPy 擅长：
    ✓ 自动优化 Agent 每一步的 Prompt
    ✓ 发现最佳的 few-shot 示例
    ✓ 结构化输出的质量保证

  LangChain 擅长：
    ✓ Agent 编排（LangGraph 的图式工作流）
    ✓ 工具集成（MCP、向量数据库、API）
    ✓ 生态系统（社区、文档、第三方集成）

  两者配合：
    用 LangGraph 搭建 Agent 的执行流程
    用 DSPy 优化每个 LLM 调用节点的 Prompt

面试中这样描述会非常加分：
  "我们用 LangGraph 做 Agent 编排（控制流），
   DSPy 做 LLM 调用优化（Prompt 层），
   两者解耦，各司其职。"


22.6 DSPy 把 Agent 开发变成「数据驱动」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统 Agent 开发流程：
  Prompt 不好 → 看结果 → 凭直觉改 → 再试

DSPy 的 Agent 开发流程：
  写代码（Module）→ 准备评测集 → 运行 Optimizer → 自动优化完成

类比：
  传统方式 = 手动调参炼丹
  DSPy = 自动超参搜索（类似 Optuna 之于深度学习）

面试金句：
  "我不手动调 Prompt，我写评测指标，
   让 DSPy Optimizer 自动找出最佳 Prompt。"
"""


def make_metric(expected_key: str):
    """创建评分函数。"""
    def metric(predicted: dict, expected: dict) -> float:
        pred_val = str(predicted.get(expected_key, ""))
        exp_val = str(expected.get(expected_key, ""))
        if pred_val == exp_val:
            return 1.0
        if exp_val.lower() in pred_val.lower():
            return 0.7
        # Token 重叠率
        pred_tokens = set(pred_val.lower().split())
        exp_tokens = set(exp_val.lower().split())
        if not exp_tokens:
            return 0.0
        overlap = len(pred_tokens & exp_tokens) / len(exp_tokens)
        return min(overlap, 0.5)
    return metric


def demo_dspy_workflow():
    """演示 DSPy 的完整工作流。"""
    print("=" * 60)
    print("  DSPy 工作流演示")
    print("=" * 60)

    # Step 1: 定义 Signature
    class SentimentSignature(DSPySignature):
        text = InputField(desc="待分析文本")
        sentiment = OutputField(desc="情感: positive/negative/neutral")
        confidence = OutputField(desc="置信度 0-1")

    print("\n  📝 Signature 自动生成的 Prompt 模板:")
    print("-" * 50)
    print(SentimentSignature.to_prompt_template())

    # Step 2: 创建 Module
    module = DSPyModule("sentiment_analyzer", SentimentSignature)
    print(f"\n  🧩 创建 Module: {module.name}")

    # Step 3: 准备训练集
    trainset = [
        {
            "inputs": {"text": "这个产品太棒了"},
            "outputs": {"sentiment": "positive", "confidence": "0.95"},
        },
        {
            "inputs": {"text": "服务非常差"},
            "outputs": {"sentiment": "negative", "confidence": "0.90"},
        },
        {
            "inputs": {"text": "还行吧，一般般"},
            "outputs": {"sentiment": "neutral", "confidence": "0.80"},
        },
        {
            "inputs": {"text": "I love this!"},
            "outputs": {"sentiment": "positive", "confidence": "0.92"},
        },
    ]
    print(f"  📊 训练集: {len(trainset)} 个样本")

    # Step 4: 运行 Optimizer
    metric = make_metric("sentiment")
    optimizer = DSPyOptimizer(metric=metric)

    print(f"\n  ⚙️ Optimizer 编译中...")
    optimized = optimizer.compile(module, trainset, max_candidates=4)

    print(f"  ✅ 编译完成!")
    print(f"  成功轨迹数: {len(optimizer.candidates)}")
    print(f"  注入的 few-shot 示例数: {len(optimized.few_shot_examples)}")

    # Step 5: 测试优化后的 Module
    print(f"\n  🧪 测试优化后的 Module:")
    tests = [
        "这部电影真的很感人",
        "无聊透顶的体验",
        "价格不高不低",
    ]
    for text in tests:
        result = optimized(text=text)
        print(f"    输入: {text[:20]}... → {result}")

    # Step 6: 对比优化前后
    print(f"\n  📊 优化前后对比:")
    print(f"  优化前 Prompt 长度: {len(module.prompt)} 字符")
    print(f"  优化后 Prompt 长度: {len(optimized.optimized_prompt)} 字符")
    print(f"  Few-shot 示例: {len(module.few_shot_examples)} → {len(optimized.few_shot_examples)}")


"""
22.7 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. DSPy = Programming, Not Prompting
   - Prompt 是编译产物，不是手写字符串
   - 代码不变，Optimizer 自动适配不同 LLM

2. 三层抽象
   - Signature: 输入输出规范（替代 Prompt 字符串）
   - Module: 可组合的 AI 组件
   - Optimizer: 自动优化 Prompt 和 Few-Shot 示例

3. DSPy × Agent
   - DSPy 优化每个 LLM 调用节点
   - LangGraph 编排整体执行流程
   - 两者解耦、各司其职

面试速记：
  "DSPy 是什么？"
  → Programming, not Prompting
  → Signature 定义输入输出 → Module 组装 → Optimizer 自动优化
  → 代码不变，换模型自动适配
  → 和 LangChain 互补：DSPy 管 Prompt 优化，LangChain 管流程编排
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第22章：DSPy —— Prompt 编译式 Agent 开发              ║")
    print("║  Signature · Module · Optimizer · 数据驱动            ║")
    print("╚══════════════════════════════════════════════════════╝")

    demo_dspy_workflow()

    print("\n▶ DSPy Optimizer 类型速查")
    print("-" * 50)
    opts = [
        ("BootstrapFewShot", "自动找最佳 few-shot 示例"),
        ("MIPROv2", "自动探索指令 + 示例组合（最推荐）"),
        ("BootstrapFinetune", "为每个 Module 自动微调模型"),
        ("BetterTogether", "组合多个 Optimizer 级联优化"),
    ]
    for name, desc in opts:
        print(f"  {name:20s} → {desc}")

    print("\n▶ DSPy vs LangChain 互补关系")
    print("-" * 50)
    print("  DSPy:     管 Prompt 优化（LLM 调用的「质量层」）")
    print("  LangChain: 管流程编排（Agent 执行的「控制层」）")
    print("  最佳实践: LangGraph 编排 + DSPy 优化")

    print("\n✅ 第22章完成！")
