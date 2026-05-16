"""
第35章：数据飞轮 —— 让 Agent 越用越好
======================================

📌 本章目标：
  1. 理解数据飞轮在 Agent 系统中的核心价值
  2. 掌握从交互日志中提取训练数据的 Pipeline
  3. 学会设计「收集→标注→改进→验证」闭环
  4. 了解持续改进的工程实践

📌 面试高频点：
  - 「你怎么让 Agent 越用越好？」
  - 「数据飞轮的具体流程是什么？」
  - 「怎么区分好的反馈和噪声？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
数据飞轮 = 今天的数据 → 明天的改进 → 后天更好的数据
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


35.1 什么是数据飞轮？—— 和传统开发的本质区别
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统软件开发是「大爆炸」模式：开发 → 测试 → 发布 → 等反馈 → 手动分析 → 手动改进 → 再发布。一个改进周期可能需要数周甚至数月。

数据飞轮（Data Flywheel）是一种不同的哲学：让系统自己从交互中学习。

飞轮这个名字来自亚马逊的经典理念。在 Agent 语境下它的意思是：
  今天的用户交互数据 → 明天就能驱动系统改进 → 后天产生更好的交互数据
  → 循环往复，像飞轮一样越转越快

这和传统开发的关键区别在于「谁来做分析」：
  - 传统：人工看日志 → 人工写改进 → 人工测试 → 人工发布
  - 飞轮：系统自动采日志 → LLM 自动标注 Bad Case → 自动触发优化
           → 自动回归评测 → 自动部署

飞轮的四阶段：
  1. 采集 —— 记录每一次用户交互 + 评分 + 反馈文字
  2. 标注 —— LLM 自动判断哪些是 Bad Case、归类问题类型
  3. 改进 —— 触发对应场景的 Prompt 优化或路由调整
  4. 验证 —— 跑回归评测确认改进有效才发布

为什么这样做很有价值？因为 Agent 的「质量」不是静态的——
Prompt 改了、工具变了、用户行为变了，Agent 的表现都可能退化。
飞轮让质量监控变成自动化、持续化的过程，而不是等用户投诉才知道出问题。
"""

import json
import time
import hashlib
from collections import defaultdict
from typing import Optional


class DataFlywheel:
    """数据飞轮 —— 从日志到改进的自动化 Pipeline。"""

    def __init__(self, improvement_threshold: int = 10):
        self.logs = []
        self.improvements = []
        self.threshold = improvement_threshold

    def log_interaction(self, user_input: str, agent_output: str,
                        user_feedback: str = None,
                        rating: int = None):
        """记录一次用户交互。

        Args:
            user_input: 用户输入。
            agent_output: Agent 输出。
            user_feedback: 用户文本反馈（可选）。
            rating: 用户评分 1-5（可选）。
        """
        entry = {
            "id": hashlib.md5(
                f"{user_input}-{time.time()}".encode()
            ).hexdigest()[:12],
            "user_input": user_input,
            "agent_output": agent_output,
            "user_feedback": user_feedback,
            "rating": rating,
            "timestamp": time.time(),
        }
        self.logs.append(entry)

        # 检测是否需要触发改进
        if rating is not None and rating <= 2:
            self._check_improvement()

    def _check_improvement(self):
        """检查是否达到改进阈值。"""
        bad_count = sum(1 for log in self.logs[-50:]
                       if log.get("rating", 5) <= 2)
        if bad_count >= self.threshold:
            self._trigger_improvement(
                "低分率超标",
                f"最近 50 次交互中 {bad_count} 次低分(≤2)",
            )

    def _trigger_improvement(self, reason: str, detail: str):
        """触发一次自动改进。"""
        improvement = {
            "timestamp": time.time(),
            "reason": reason,
            "detail": detail,
            "total_interactions": len(self.logs),
            "action": "建议重新评测 + 优化对应场景的 Prompt",
        }
        self.improvements.append(improvement)

    def get_stats(self) -> dict:
        """获取飞轮统计。"""
        if not self.logs:
            return {"total": 0}

        ratings = [l["rating"] for l in self.logs
                   if l.get("rating") is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        return {
            "total_interactions": len(self.logs),
            "avg_rating": round(avg_rating, 1),
            "low_rated": sum(1 for r in ratings if r <= 2),
            "improvements_triggered": len(self.improvements),
            "latest_improvement": (
                self.improvements[-1]["reason"]
                if self.improvements else "暂无"
            ),
        }

    def export_bad_cases(self, limit: int = 10) -> list:
        """导出 Bad Case 用于分析。"""
        bad = [l for l in self.logs if l.get("rating", 5) <= 2]
        bad.sort(key=lambda x: x.get("rating", 5))
        return bad[:limit]


def demo_flywheel():
    print("=" * 60)
    print("  数据飞轮演示")
    print("=" * 60)

    fw = DataFlywheel(improvement_threshold=3)

    # 模拟用户交互
    interactions = [
        ("天气查询", "晴天25°C", None, 5),
        ("订单查询", "已发货", None, 4),
        ("天气查询", "没有找到", "回答错误", 1),
        ("退货咨询", "7个工作日", None, 5),
        ("订单查询", "查不到", "订单号错了", 2),
        ("物流查询", "超时", "太慢了", 1),
        ("天气查询", "错误", "不对", 1),
    ]

    for user, output, fb, rating in interactions:
        fw.log_interaction(user, output, fb, rating)
        icon = "⭐" * rating if rating else "—"
        print(f"  [{icon}] {user} → {output[:20]}... "
              + (f"反馈: {fb}" if fb else ""))

    stats = fw.stats()
    print(f"\n  📊 飞轮统计:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print(f"\n  🐛 Bad Case ({len(fw.export_bad_cases())} 条):")
    for case in fw.export_bad_cases():
        print(f"    [{case['rating']}★] {case['user_input']} "
              f"→ {case['agent_output'][:30]}...")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第35章：数据飞轮                                      ║")
    print("║  交互采集 · Bad Case 识别 · 自动触发改进              ║")
    print("╚══════════════════════════════════════════════════════╝")
    demo_flywheel()
    print("\n▶ 飞轮四阶段")
    print("-" * 50)
    for stage, desc in [
        ("1. 采集", "记录所有交互 + 用户反馈"),
        ("2. 标注", "LLM 自动标注 Bad Case"),
        ("3. 改进", "触发 Prompt 优化 or 路由调整"),
        ("4. 验证", "回归评测 → 确认改进 → 发布"),
    ]:
        print(f"  {stage:8s} → {desc}")
    print("\n✅ 第35章完成！")
