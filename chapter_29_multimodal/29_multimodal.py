"""
第29章：Multi-Modal Agent —— 让 Agent 看懂世界
================================================

内容核对：2026-08-01
说明：标注为模拟的实现与数值用于讲解概念，不代表真实 SDK、协议或基准结果。

📌 本章目标：
  1. 理解多模态 Agent 的核心架构（视觉+文本联合推理）
  2. 掌握 Responses API 的图像输入格式，并理解 provider 差异
  3. 实现图片分析 Agent 和视频帧提取 Agent
  4. 理解视觉上下文对 Agent 决策的影响

📌 面试高频点：
  - 「你的 Agent 能处理图片吗？怎么架构的？」
  - 「多模态和纯文本 Agent 的区别在哪？」
  - 「视觉 Token 的成本怎么算？」

多模态模型、输入限制和计费方式变化很快；示例模型通过环境变量配置，
运行前查看目标 provider 的当前模型与视觉 token 文档。


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
  4. 图片 token/成本取决于模型、尺寸、detail 与 provider 计算规则

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

使用 OpenAI Responses API 发送图片：

  response = client.responses.create(
      model=os.environ["LLM_MODEL"],
      input=[{
          "role": "user",
          "content": [
              {"type": "input_text", "text": "这张图片里有什么？"},
              {"type": "input_image",
               "image_url": f"data:image/jpeg;base64,{base64_str}"},
          ],
      }],
  )
  print(response.output_text)

Tool Calling + 多模态：
  Agent 可以先「看」一张截图，然后决定调用哪个工具。
  这就是 Computer Use 的基础（Ch17 讲过截图-动作循环）。


29.3 图片分析 Agent 实现
━━━━━━━━━━━━━━━━━━━━━━━━

下面实现一个 ImageAnalysisAgent 类，模拟多模态 Agent 的完整决策流程。
虽然这里用规则模拟替代了真实视觉模型调用，但保留了核心的三步
Pipeline：视觉理解 → 工具决策 → 生成回答。这套 Pipeline 的架构和真实
系统相似，但生产替换还需处理媒体类型、尺寸限制、拒绝、隐私和结果验证。
"""

import base64
import json
import time
import os
from dataclasses import dataclass
from typing import Optional


class ImageAnalysisAgent:
    """多模态 Agent —— 模拟图片分析 + 工具调用。

    真实实现应接入当前支持图像输入的模型；OpenAI 示例默认使用
    GPT-5.6 系列，并通过环境变量固定具体模型 ID。
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

设计多模态 Agent 不只是「把图片传进去」，需要像精算师一样
计算成本和可靠性。以下 4 条原则是工业界反复验证的经验。

1. Token 预算管理
   图片计费受模型、尺寸、detail 和 provider 规则影响；必须读取响应 usage
   与实际账单建立每类图片的成本分布，而不是用固定“每张图 token”估算。
   
   对策：
    - 在不损失任务所需细节的前提下缩放/裁剪，并用评测确定尺寸
    - 按请求预算限制图片数量、像素和总输入
    - 用预算管理系统（Ch28）限制单用户每日视觉 Token
   
   为什么这样设计？因为 LLM 对图片按像素区域计费，不是按「信息量」。
   一张模糊的照片和一张高清照片，对人类来说信息量可能相同，
   但对 LLM 来说 Token 消耗差几十倍。

2. 视觉信息优先级 —— 分层分析策略
   像医生看病一样：先看 X 光片（缩略图）→ 再看 CT（高清细节）。
   低成本的预览阶段如果已经能得到足够信息，就不必进入高成本阶段。
   
   具体实现：先传 512px 缩略图做初步判断（廉价），如果检测到
   关键信息（如文字、人脸、二维码），再传原图做精细分析（昂贵）。
   
   分层策略是否降低成本且保持准确率，取决于图片分布；用随机抽样和困难集
   比较单阶段与两阶段方案，不假设固定节省比例。

3. 视频处理策略 —— 不要逐帧分析
   30fps 的视频每秒 30 帧，逐帧发送通常浪费预算。候选做法是固定间隔采样、
   场景切换检测、字幕/音频对齐，再按任务做聚合推理。
   
   更高级的做法：用轻量模型检测场景切换点，只在画面变化大时
   抽取帧。最终帧数由事件密度和漏检成本决定。
   
   这就是现实工程中的取舍：不是「分析得越多越好」，
   而是「在成本可接受范围内获取最大信息量」。

4. 错误处理 —— 视觉模型比你想象的更容易出错
   和文本不同，视觉理解有天然的模糊性。同一个物体在不同角度、
   光照下，LLM 可能给出不同判断。
   
   关键信息（如发票金额）应使用 OCR/规则/数据库交叉校验或人工确认；
   同一模型重复两次结果一致并不代表事实正确。
   
   模糊图片不应直接猜测，而应请求用户提供更清晰的图片——
   这看似简单，但很多 Agent 产品忽略了这一点，导致严重错误。


29.5 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. 多模态 Agent = 文本推理 + 视觉理解 + 工具联动
2. 视觉预算与文本不能用跨模型固定比例比较，需要读取 usage 和账单
3. Computer Use 是多模态的一种高风险应用，不是所有多模态 Agent 的终点

面试速记：
  「多模态 Agent 怎么做的？」
  → 图片转 base64 → 嵌入 messages → LLM 视觉+文本联合推理
  → 分析结果驱动工具调用
  → 按模型、尺寸与 detail 记录 usage，并验证关键字段
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
