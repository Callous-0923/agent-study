"""
第29章：Multi-Modal Agent —— 让 Agent 看懂世界
================================================

📌 本章目标：
  1. 理解多模态 Agent 的核心架构（视觉+文本联合推理）
  2. 掌握 GPT-4o / Claude 3.5 / Gemini 2.5 的多模态 API
  3. 实现图片分析 Agent 和视频帧提取 Agent
  4. 理解视觉上下文对 Agent 决策的影响

📌 面试高频点：
  - 「你的 Agent 能处理图片吗？怎么架构的？」
  - 「多模态和纯文本 Agent 的区别在哪？」
  - 「视觉 Token 的成本怎么算？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2025 全系多模态：GPT-4o / Claude 3.5 / Gemini 2.5 Flash
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


29.1 多模态 Agent 的架构本质
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

纯文本 Agent：
  user → LLM(text) → tools → LLM(text) → response

多模态 Agent：
  user + image/audio/video → LLM(vision+text) → tools → response

关键区别：
  1. 输入不只是文本，还有 base64 编码的图片/音频
  2. 视觉信息作为额外的「上下文通道」
  3. LLM 内部做视觉编码 → 和文本 token 融合
  4. 视觉 Token 消耗远大于文本（一张图的成本 ≈ 500-1500 字）

多模态 Agent 的三种模式：

┌──────────────────┬──────────────────────────────────────┐
│ 模式               │              说明                    │
├──────────────────┼──────────────────────────────────────┤
│ 直接视觉推理       │ 图片→LLM 直接理解内容                  │
│ 视觉+工具联动      │ 先从图片提取信息→调用工具→结合结果      │
│ 视频流分析         │ 逐帧提取→聚合分析→时间序列推理         │
└──────────────────┴──────────────────────────────────────┘


29.2 OpenAI 多模态 API 速览
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

发送图片给 GPT-4o：

  response = client.chat.completions.create(
      model="gpt-4o",
      messages=[{
          "role": "user",
          "content": [
              {"type": "text", "text": "这张图片里有什么？"},
              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}},
          ],
      }],
  )

Tool Calling + 多模态：
  Agent 可以先「看」一张截图，然后决定调用哪个工具。
  这就是 Computer Use 的基础（Ch17 讲过截图-动作循环）。


29.3 图片分析 Agent 实现
━━━━━━━━━━━━━━━━━━━━━━━━
"""

import base64
import json
import time
import os
from dataclasses import dataclass
from typing import Optional


class ImageAnalysisAgent:
    """多模态 Agent —— 模拟图片分析 + 工具调用。

    真实实现接入 GPT-4o Vision 或 Claude 3.5 Sonnet。
    这里用规则模拟来展示多模态 Agent 的决策流程。
    """

    def __init__(self):
        self.tool_log = []

    def analyze(self, image_description: str,
                question: str) -> dict:
        """分析图片并回答问题。

        Args:
            image_description: 图片内容描述（模拟视觉分析结果）。
            question: 用户提问。

        Returns:
            包含分析结果的字典。
        """
        # 第1步：视觉理解
        objects = self._detect_objects(image_description)
        text_in_image = self._extract_text(image_description)

        # 第2步：工具决策
        tools_needed = []
        if "表格" in image_description or "数据" in image_description:
            tools_needed.append("extract_table_data")
        if any(w in question for w in ["翻译", "translate"]):
            tools_needed.append("translate")

        # 第3步：生成回答
        answer = self._generate_answer(objects, text_in_image,
                                       question, tools_needed)

        self.tool_log.append({
            "image_desc": image_description[:80],
            "question": question,
            "objects": objects,
            "tools": tools_needed,
        })

        return {
            "objects_found": len(objects),
            "text_detected": text_in_image[:100],
            "tools_invoked": tools_needed,
            "answer": answer,
        }

    def _detect_objects(self, desc: str) -> list[str]:
        objects = []
        obj_keywords = ["人", "车", "猫", "狗", "电脑", "手机",
                        "书", "杯子", "建筑", "道路", "屏幕",
                        "图表", "按钮", "文字", "二维码"]
        for obj in obj_keywords:
            if obj in desc:
                objects.append(obj)
        return objects or ["未知物体"]

    def _extract_text(self, desc: str) -> str:
        # 模拟 OCR
        if "文字" in desc or "文本" in desc:
            return "[OCR结果] 检测到文字内容"
        return "[未检测到文字]"

    def _generate_answer(self, objects: list, text: str,
                         question: str, tools: list) -> str:
        obj_list = "、".join(objects)
        base = f"图片中包含: {obj_list}。{text}。"
        if tools:
            base += f" 已调用工具: {', '.join(tools)}。"
        base += f" 针对您的问题「{question}」，我的回答是："
        base += "根据图片内容分析，这是一个模拟的多模态 Agent 回答。"
        return base


"""
29.4 多模态 Agent 的设计原则
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Token 预算管理
   一张 1080p 图片 ≈ 1500-5000 tokens（取决于压缩）
   一次 GPT-4o 视觉调用可能花费 $0.01-0.05
   → 控制图片数量和质量

2. 视觉信息优先级
   先看缩略图（低成本） → 再看高清细节（高成本）
   → 分层视觉分析

3. 视频处理策略
   不要逐帧分析（成本爆炸）
   关键帧提取（每秒1帧） + 聚合推理

4. 错误处理
   视觉模型可能误读 → 二次确认机制
   模糊图片 → 请求用户提供更清晰的图片


29.5 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. 多模态 Agent = 文本推理 + 视觉理解 + 工具联动
2. 视觉 Token 成本远高于文本，需要预算管理
3. Computer Use 是多模态 Agent 的终极形态

面试速记：
  「多模态 Agent 怎么做的？」
  → 图片转 base64 → 嵌入 messages → LLM 视觉+文本联合推理
  → 分析结果驱动工具调用
  → 注意 Token 成本（图片 ≈ 500-5000 tokens）
"""


def demo_multimodal():
    print("=" * 60)
    print("  多模态 Agent 演示")
    print("=" * 60)

    agent = ImageAnalysisAgent()

    scenarios = [
        ("屏幕截图显示了一个错误对话框，文字提示'数据库连接失败'",
         "该怎么解决这个错误？"),
        ("图片中有一张表格，包含销售额数据，文字包括'Q1: 100万, Q2: 150万'",
         "Q3的目标应该设为多少？"),
        ("照片里有一本英文书籍封面，文本是'Designing Data-Intensive Applications'",
         "请翻译这本书的书名"),
    ]

    for desc, question in scenarios:
        result = agent.analyze(desc, question)
        print(f"\n  📷 图片: {desc[:50]}...")
        print(f"  ❓ 问题: {question}")
        print(f"  🔍 检测到 {result['objects_found']} 个物体")
        print(f"  🔧 使用工具: {result['tools_invoked'] or '无需工具'}")
        print(f"  💬 回答: {result['answer'][:80]}...")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第29章：Multi-Modal Agent                             ║")
    print("║  视觉+文本联合推理 · Token 预算 · 视频帧分析           ║")
    print("╚══════════════════════════════════════════════════════╝")
    demo_multimodal()
    print("\n▶ 多模态三模式")
    print("-" * 50)
    for mode, desc in [
        ("直接视觉推理", "图片→LLM 直接理解"),
        ("视觉+工具联动", "提取信息→调工具→结合"),
        ("视频流分析", "逐帧提取→聚合推理"),
    ]:
        print(f"  {mode:16s} → {desc}")
    print("\n✅ 第29章完成！")
