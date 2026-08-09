"""
第34章：面向工具调用的模型微调
==============================

内容核对：2026-08-01
说明：标注为模拟的实现与数值用于讲解概念，不代表真实 SDK、协议或基准结果。

📌 本章目标：
  1. 理解什么时候值得微调一个 Agent 专用模型
  2. 掌握 LoRA 微调的基本概念和流程
  3. 了解 Function Calling 微调的数据准备方法
  4. 用离线评测、线上护栏和总拥有成本判断是否上线

📌 面试高频点：
  - 「为什么不直接用大模型，还要微调？」
  - 「LoRA 是什么？怎么用在 Agent 上？」
  - 「微调后的小模型能直接替代通用大模型吗？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
以下 JSONL 生成器是教学示例。真实训练格式、可微调模型和工具调用字段必须以所选
平台的当前文档为准；模型能力和价格不应硬编码进课程。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


34.1 先证明“值得微调”
━━━━━━━━━━━━━━━━━━━━━━━

微调可能改善稳定格式、领域风格、工具选择或参数提取，但不会自动补齐知识、权限控制，
也不能保证小模型获得基础模型没有的推理能力。先按顺序检查：
  1. Prompt、结构化输出、检索和工具描述是否已经优化？
  2. 是否有来自真实失败模式、经过脱敏和授权的高质量样本？
  3. 是否存在固定的训练集、验证集和完全隔离的测试集？
  4. 基线模型在任务成功率、参数准确性和安全约束上差在哪里？
  5. 训练、托管、GPU 利用率、监控、数据治理和回滚的总成本是否更低？

不存在适用于所有项目的调用量阈值或准确率差值。用自己的流量分布和错误成本做
盈亏平衡分析；本地推理也有硬件、运维、能耗和闲置成本，并非“接近零”。


34.2 Function Calling 微调数据格式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

每条训练数据是一个 (user_message, tool_call_or_answer) 对：

  {
    "messages": [
      {"role": "system", "content": "你是客服 Agent..."},
      {"role": "user",   "content": "帮我查订单 #12345"},
      {"role": "assistant", "content": null,
       "tool_calls": [{
         "function": {"name": "query_order", "arguments": "{\"order_id\":\"12345\"}"}
       }]}
    ]
  }

样本量没有通用的“最少、理想或过拟合”分界。先保证覆盖关键意图、边界条件、拒绝样本、
工具失败和无需调用工具的反例，再通过学习曲线判断新增数据是否仍有价值。重复或泄漏到
测试集的数据，即使数量很大，也不会带来可信的泛化结论。


34.3 LoRA 微调概述
━━━━━━━━━━━━━━━━━━

LoRA = Low-Rank Adaptation

原理：
  不修改原模型权重，训练时插入低秩矩阵。
  冻结基础权重，在选定层训练低秩适配器；可训练参数量由 rank、目标层和模型结构决定。
  它通常降低训练显存与存储需求，但具体加速或成本必须实测，不能套用固定倍数。

适合 Agent 的场景：
  ✓ 工具选择（classification 任务）
  ✓ 参数提取（从自然语言到 JSON）
  ✓ 意图识别（routing 的入口）


34.4 候选方案如何比较
━━━━━━━━━━━━━━━━━━━━━━━

对“基础模型”“微调模型”“路由组合”使用同一数据切片和同一工具环境，至少比较：
  - 端到端任务成功率，而非只看分类准确率；
  - 工具名称、参数与调用时机的准确性；
  - 高风险误操作、越权、提示注入与拒绝行为；
  - p50/p95 延迟、输入输出 tokens、GPU 利用率和总拥有成本；
  - 分布外问题、工具错误和模型升级后的退化。

只有通过隔离测试集与安全门槛的候选版本才能进入小流量 canary。路由比例来自线上评测，
不是预先规定的常数；保留回退模型和版本化数据，以便快速回滚。
"""

import json


class FineTuneDataGenerator:
    """Function Calling 微调数据生成器。"""

    def __init__(self):
        self.samples = []

    def add_sample(self, system_prompt: str, user_msg: str,
                   tool_name: str = None,
                   tool_args: dict = None,
                   direct_answer: str = None):
        """添加一条微调数据。

        既可以是 tool_call 样本，也可以是 direct_answer 样本。
        """
        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }

        if tool_name and tool_args:
            sample["messages"].append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args, ensure_ascii=False),
                    }
                }],
            })
        elif direct_answer:
            sample["messages"].append({
                "role": "assistant",
                "content": direct_answer,
            })

        self.samples.append(sample)

    def export_jsonl(self, filepath: str = None) -> str:
        """导出为 JSONL 格式。

        Returns:
            JSONL 字符串。
        """
        lines = [json.dumps(s, ensure_ascii=False) for s in self.samples]
        return "\n".join(lines)

    def stats(self) -> dict:
        cnt = sum(1 for s in self.samples
                  if "tool_calls" in s["messages"][-1])
        return {
            "total": len(self.samples),
            "tool_call_samples": cnt,
            "direct_answer_samples": len(self.samples) - cnt,
        }


def demo_finetune():
    print("=" * 60)
    print("  Function Calling 微调数据准备")
    print("=" * 60)

    gen = FineTuneDataGenerator()
    system = "你是客服 Agent，负责查询订单和物流。"

    # Tool call 样本
    gen.add_sample(system, "帮我查订单 ORD-001 的状态",
                   "query_order", {"order_id": "ORD-001"})
    gen.add_sample(system, "查物流：SF1234567890",
                   "query_logistics", {"tracking_no": "SF1234567890"})
    gen.add_sample(system, "订单 A-999 到哪了？",
                   "query_order", {"order_id": "A-999"})

    # Direct answer 样本
    gen.add_sample(system, "你们几点上班？",
                   direct_answer="我们的客服时间是 9:00-18:00。")
    gen.add_sample(system, "退货要几天？",
                   direct_answer="退款在 7 个工作日内到账。")

    stats = gen.stats()
    print(f"  总样本: {stats['total']}")
    print(f"  Tool Call 样本: {stats['tool_call_samples']}")
    print(f"  直接回答样本: {stats['direct_answer_samples']}")

    jsonl = gen.export_jsonl()
    print(f"\n  JSONL 预览 (头 2 条):")
    for i, line in enumerate(jsonl.split("\n")[:2]):
        print(f"  [{i}] {line[:120]}...")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第34章：模型微调 for Function Calling                 ║")
    print("║  LoRA · 微调数据准备 · 成本收益对比                    ║")
    print("╚══════════════════════════════════════════════════════╝")
    demo_finetune()
    print("\n▶ 候选方案的评测维度")
    print("-" * 50)
    for dimension in [
        "端到端任务成功率与工具参数准确性",
        "高风险误操作、越权与拒绝行为",
        "p50/p95 延迟、tokens 与总拥有成本",
        "分布外、工具错误和模型升级后的鲁棒性",
    ]:
        print(f"  - {dimension}")
    print("\n✅ 第34章完成！")
