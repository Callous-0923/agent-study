"""
第35章：数据飞轮——以评测和治理驱动持续改进
==========================================

内容核对：2026-08-01
说明：标注为模拟的实现与数值用于讲解概念，不代表真实 SDK、协议或基准结果。

📌 本章目标：
  1. 理解数据飞轮在 Agent 系统中的核心价值
  2. 掌握从合规交互数据中提取评测候选样本的 Pipeline
  3. 学会设计「收集→标注→改进→验证」闭环
  4. 了解持续改进的工程实践

📌 面试高频点：
  - 「你怎么让 Agent 越用越好？」
  - 「数据飞轮的具体流程是什么？」
  - 「怎么区分好的反馈和噪声？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
数据飞轮 = 可治理的数据 → 候选改进 → 隔离评测 → 受控发布
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


35.1 什么是数据飞轮？—— 和传统开发的本质区别
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

数据飞轮（Data Flywheel）不是让线上 Agent 无监督地改写自己，而是把生产信号稳定地转化为
可审计的候选改进。自动化可以帮助采样、聚类、去重和初步标注，但 LLM 评分本身也会有偏差；
高风险数据和发布决策仍需要人工责任人及明确门禁。

飞轮的四阶段：
  1. 采集 —— 依同意、用途和保留期采集必要字段；脱敏、访问控制并记录数据血缘
  2. 标注 —— 聚类 Bad Case，结合用户反馈、规则、模型裁判与人工复核
  3. 改进 —— 生成版本化的 Prompt、检索、工具、路由或训练数据候选
  4. 验证 —— 冻结测试集，执行质量、安全和成本回归，再经审批、canary 与回滚发布

为什么这样做很有价值？因为 Agent 的「质量」不是静态的——
Prompt 改了、工具变了、用户行为变了，Agent 的表现都可能退化。
飞轮让质量监控变成持续过程，但“用户给低分”等代理信号不等于真实错误。还要校正选择偏差、
反馈操纵、隐私泄露、训练与测试污染，以及优化单一指标导致的反常行为。
"""

import time
import hashlib


class DataFlywheel:
    """教学模拟：从已脱敏反馈生成“待评测改进建议”，不会自动修改或部署系统。"""

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

        # 真实系统应在写入前完成同意检查、数据最小化和脱敏。
        # 此处只检测是否需要生成待评测建议。
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
        """创建一条候选改进建议；不会自动修改 Prompt 或部署。"""
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
            "improvement_candidates": len(self.improvements),
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

    stats = fw.get_stats()
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
    print("║  合规采集 · Bad Case 识别 · 候选改进 · 受控验证        ║")
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
