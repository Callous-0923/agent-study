"""
第34章：模型微调 for Function Calling —— 用小模型做 Agent
=============================================================

📌 本章目标：
  1. 理解什么时候值得微调一个 Agent 专用模型
  2. 掌握 LoRA 微调的基本概念和流程
  3. 了解 Function Calling 微调的数据准备方法
  4. 对比微调 vs 路由的成本收益

📌 面试高频点：
  - 「为什么不直接用大模型，还要微调？」
  - 「LoRA 是什么？怎么用在 Agent 上？」
  - 「微调后的小模型能替代 GPT-4o 吗？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
综合 LoRA 论文 + OpenAI Fine-Tuning API + 业界实践
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


34.1 为什么微调 Agent？
━━━━━━━━━━━━━━━━━━━━☆

场景：客服 Agent，每天 100 万次简单查询。
  全用 GPT-4o → $2500/天
  全用微调的 Llama-3B → $50/天
  → 微调小模型做分类+简单问答，大模型兜底复杂问题

微调的价值：
  ✓ 成本降低 50-100x（小模型 vs 大模型）
  ✓ 延迟降低 10-50x（本地推理 vs API 调用）
  ✓ 专用场景准确率可能超过通用大模型

微调的成本：
  ✗ 需要标注数据（100-1000 条）
  ✗ 需要 GPU（但可以用 LoRA + Colab Free Tier）
  ✗ 需要维护模型版本


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

数据量建议：
  - 最少 50-100 条（少量微调）
  - 理想 500-2000 条（完整微调）
  - 超过 5000 条（可能过拟合）


34.3 LoRA 微调概述
━━━━━━━━━━━━━━━━━━

LoRA = Low-Rank Adaptation

原理：
  不修改原模型权重，训练时插入低秩矩阵。
  原模型参数: 1B
  LoRA 参数:   1M (约 0.1%)
  → 训练成本降低 1000x

适合 Agent 的场景：
  ✓ 工具选择（classification 任务）
  ✓ 参数提取（从自然语言到 JSON）
  ✓ 意图识别（routing 的入口）


34.4 微调 vs 路由 vs 全用大模型
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────┬──────────┬──────────┬──────────┐
│                │ 成本/次   │  延迟     │  准确率   │
├──────────────┼──────────┼──────────┼──────────┤
│ 全用 GPT-4o    │ $0.01    │ 2s       │ 95%      │
│ 微调 Llama-3B  │ $0.0002  │ 0.1s     │ 90%      │
│ 路由(90%小模型)  │ $0.0012  │ 0.4s     │ 94%      │
└──────────────┴──────────┴──────────┴──────────┘

最佳实践：路由为主 + 微调为辅
  - 80-90% 查询 → 微调小模型
  - 10-20% 查询 → GPT-4o
  - 综合成本降低 85%，准确率损失 < 2%
"""

import json
import time
from collections import defaultdict


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
    print("\n▶ 微调 vs 路由 vs 全用大模型")
    print("-" * 50)
    for name, cost, lat, acc in [
        ("全用 GPT-4o", "$0.01/次", "2s", "95%"),
        ("微调 Llama-3B", "$0.0002/次", "0.1s", "90%"),
        ("路由(90%小模型)", "$0.0012/次", "0.4s", "94%"),
    ]:
        print(f"  {name:18s} | {cost:12s} | {lat:6s} | {acc}")
    print("\n✅ 第34章完成！")
