"""
第17章：Computer Use + GUI Agent —— AI 操控电脑
==================================================

📌 本章目标：
  1. 理解 Computer Use / GUI Agent 的核心原理（Screenshot-Action Loop）
  2. 掌握 Anthropic Computer Use 和 OpenAI CUA 的架构差异
  3. 理解像素坐标的计算与视觉定位机制
  4. 了解 Browser Use 等开源方案
  5. 认识安全沙箱的必要性和实现方式

📌 面试高频点：
  - Computer Use 的原理是什么？和传统 API 调用有什么区别？
  - Screenshot-Action Loop 的每一步做了什么？
  - OpenAI CUA 和 Anthropic Computer Use 的架构差异？
  - Computer Use 的安全风险有哪些？怎么防护？


17.1 为什么需要 Computer Use？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统 Agent 的局限：
  Agent 只能调 API → 但世界上绝大多数软件没有 API！
  - 企业内部的遗留系统
  - 桌面软件（Photoshop、Excel）
  - 图形化界面的 SaaS 工具

Computer Use 的突破：
  Agent 不再需要对方提供 API
  它直接「看屏幕 → 分析画面 → 控制鼠标键盘」
  就像人类一样和任何软件交互

类比：
  传统 Agent = 只能打电话的人（必须对方有号码）
  Computer Use = 能走进办公室的人（可以和任何人面对面交流）


17.2 Screenshot-Action Loop —— 核心循环
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是所有 Computer Use 系统的底层逻辑：

  ┌──────────────────────────────────────────────────┐
  │                                                    │
  │  1. 📸 Screenshot: 截取当前屏幕画面                 │
  │         │                                          │
  │         ▼                                          │
  │  2. 👁️ Analyze: LLM 视觉分析画面                    │
  │     - 识别窗口、按钮、文本框                        │
  │     - 读取屏幕上的文字内容                          │
  │     - 理解当前界面的状态                            │
  │         │                                          │
  │         ▼                                          │
  │  3. 🤔 Decide: 决定下一步操作                       │
  │     - 应该点击哪里？                                │
  │     - 应该输入什么？                                │
  │     - 是否需要滚动？                                │
  │         │                                          │
  │         ▼                                          │
  │  4. 🖱️ Execute: 执行操作                            │
  │     - mouse_move(x, y)                             │
  │     - left_click()                                 │
  │     - type("文本")                                 │
  │     - scroll(direction)                            │
  │     - key_press("Enter")                           │
  │         │                                          │
  │         ▼                                          │
  │     回到步骤 1（直到任务完成）                       │
  │                                                    │
  └──────────────────────────────────────────────────┘

关键挑战：像素坐标的精确计算

  问题：LLM 需要输出 「点击 (450, 200)」这样的坐标
  但 LLM 是文本模型，不理解像素

  Anthropic 的解决方案（训练阶段）：
    专门训练 Claude 精确计数像素的能力
    "Training Claude to count pixels accurately was critical.
     Without this skill, the model finds it difficult to give mouse commands."

  实操中的坐标系统：
    - 截图尺寸通常是 1280x800 或 1920x1080
    - LLM 返回的坐标需要缩放到实际屏幕分辨率
    - 返回格式：(x_pct, y_pct) 百分比比绝对像素更稳健


17.3 Anthropic Computer Use vs OpenAI CUA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────┬─────────────────────────┬─────────────────────────┐
│     维度      │  Anthropic Computer Use  │   OpenAI CUA            │
├──────────────┼─────────────────────────┼─────────────────────────┤
│ 发布时间      │ 2024.10 (API)            │ 2025.01 (Operator)       │
│             │ 2025.10 (正式发布)        │                         │
│ 操作范围      │ 整个操作系统              │ 浏览器内（虚拟浏览器）    │
│ 模型          │ Claude 3.5+ Sonnet      │ GPT-4o (CUA 微调版)     │
│ 环境          │ 用户真实桌面/Docker      │ 安全的虚拟浏览器环境      │
│ 安全性        │ 依赖使用者自行沙箱         │ 平台内置安全隔离         │
│ 动作类型      │ 鼠标+键盘+截图            │ 浏览器操作（点击/输入/滚动）│
│ 成本          │ 截图Token昂贵            │ 浏览器操作Token消耗较低   │
│ Benchmark    │ OSWorld 14.9%            │ 未公布独立评分           │
└──────────────┴─────────────────────────┴─────────────────────────┘

Anthropic 的设计哲学：
  「给 Claude 真实的电脑，让它按人类的方式工作」

OpenAI 的设计哲学：
  「给 GPT-4o 一个安全沙箱，专注于 Web 任务」

选型建议：
  - 需要控制桌面软件 → Anthropic Computer Use
  - 只需要浏览器操作 → OpenAI CUA
  - 想要完全控制 → Anthropic + Docker 沙箱


17.4 性能数据与局限
━━━━━━━━━━━━━━━━━━━

OSWorld Benchmark 成绩：
  ┌──────────────────┬───────────┐
  │      系统         │   得分     │
  ├──────────────────┼───────────┤
  │ 人类              │   75.0%   │
  │ Claude 3.5 Sonnet │   14.9%   │
  │ GPT-4V            │    7.8%   │
  └──────────────────┴───────────┘

  → Claude 翻倍了前最好成绩，但离人类还很远

延迟：
  - 每个动作 3-4 秒（截取→分析→执行）
  - 10步任务 = 30-40秒
  - 对比 Selenium 的 0.1秒/步，差距巨大

成本：
  - 每张 1080p 截图消耗约 1500 tokens
  - 每分钟成本约 $0.10-0.30
  - 对比 API 调用的 $0.001/分钟，贵 100 倍

当前定位（2025-2026）：
  → 不是 Selenium 的替代品
  → 适合「API 无法覆盖的长尾场景」
  → 适合「快速原型验证」
  → 生产级自动化仍需传统方案


17.5 安全沙箱 —— 必须学！
━━━━━━━━━━━━━━━━━━━━━━━━━━━

给 AI 鼠标键盘的权限 = 极高的安全风险：
  ✗ 读取屏幕上的密码
  ✗ 复制敏感数据
  ✗ 误操作删除文件
  ✗ Prompt Injection 利用 AI 执行危险命令

安全措施（必须！）：

  1. Docker 容器隔离
     docker run -d \
       --security-opt=no-new-privileges \
       --cap-drop=ALL \
       --network=none \
       --read-only \
       computer-use-sandbox

  2. 操作系统级隔离
     - 非管理员用户
     - 只读挂载关键目录
     - 网络访问白名单

  3. 操作确认（Human-in-the-Loop）
     - 危险操作需用户确认
     - 大额交易/删除文件 → 二次确认

  4. 审计日志
     - 记录每一次鼠标点击和键盘输入
     - 可追溯所有操作

Anthropic 官方建议：
  "Always sandbox in Docker containers with limited permissions.
   Never run with admin privileges."


17.6 模拟 Computer Use Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━

下面实现一个 ComputerUseAgent 模拟器，模拟 Claude 的「截图→分析→动作」
核心循环。真实的 Computer Use 每次迭代需要传入 desktop screenshot 的 base64
给 Claude Vision，这里用描述文字替代。关键保留了三阶段：环境感知、动作决策、
执行反馈，以及坐标计算的容错逻辑。
"""

import time
import json
from typing import Optional


class VirtualScreen:
    """模拟计算机屏幕 —— 用作 Computer Use 的目标环境。"""

    def __init__(self, width: int = 1280, height: int = 800):
        self.width = width
        self.height = height
        self.elements = {}  # 屏幕上的 UI 元素 {name: (x, y, w, h)}

    def add_element(self, name: str, x: int, y: int,
                    w: int, h: int, text: str = ""):
        """添加一个 UI 元素（按钮/输入框/文本）。"""
        self.elements[name] = {
            "x": x, "y": y, "w": w, "h": h, "text": text,
        }

    def find_element_at(self, click_x: int, click_y: int) -> Optional[str]:
        """根据坐标查找被点击的元素。"""
        for name, elem in self.elements.items():
            if (elem["x"] <= click_x <= elem["x"] + elem["w"] and
                    elem["y"] <= click_y <= elem["y"] + elem["h"]):
                return name
        return None

    def describe(self) -> str:
        """生成屏幕描述（模拟 LLM 视觉分析的结果）。"""
        lines = [f"屏幕分辨率: {self.width}x{self.height}"]
        for name, elem in self.elements.items():
            lines.append(
                f"  [{name}] 位置({elem['x']},{elem['y']}) "
                f"大小({elem['w']}x{elem['h']}) 文本:「{elem['text']}」"
            )
        return "\n".join(lines)


class ComputerUseAgent:
    """Computer Use Agent —— 模拟完整的 Screenshot-Action Loop。

    核心循环：
      while not done:
          screenshot → analyze → decide → execute → observe
    """

    def __init__(self):
        self.action_log = []
        self.max_actions = 10

    def analyze_screen(self, screen: VirtualScreen) -> dict:
        """模拟 LLM 分析屏幕截图的结果。

        真实实现中，这一步由 Claude 的视觉能力完成：
          - 上传截图（base64）到 Claude API
          - Claude 返回对界面元素的分析和下一步操作的建议

        Args:
            screen: 虚拟屏幕对象。

        Returns:
            分析结果。
        """
        description = screen.describe()
        return {
            "resolution": f"{screen.width}x{screen.height}",
            "elements_found": len(screen.elements),
            "description": description,
        }

    def decide_action(self, task: str,
                      screen: VirtualScreen) -> Optional[dict]:
        """模拟 LLM 决定的下一步操作。

        真实实现中，Claude 返回结构化的工具调用：
          tool: "computer"
          action: {"type": "left_click", "x": 450, "y": 200}

        这里用简化的规则模拟：
          根据任务关键词匹配屏幕上的元素。

        Args:
            task: 任务描述。
            screen: 当前屏幕。

        Returns:
            操作指令字典。
        """
        for name, elem in screen.elements.items():
            if name.lower() in task.lower():
                # 计算元素中心坐标
                cx = elem["x"] + elem["w"] // 2
                cy = elem["y"] + elem["h"] // 2
                return {
                    "type": "left_click",
                    "x": cx,
                    "y": cy,
                    "target": name,
                    "reasoning": f"找到了匹配元素 '{name}'，点击其中心({cx}, {cy})",
                }

        return {
            "type": "type_text",
            "text": task,
            "target": "search_box",
            "reasoning": "未找到匹配按钮，尝试搜索",
        }

    def execute_action(self, action: dict,
                       screen: VirtualScreen) -> dict:
        """执行操作并返回结果。

        Args:
            action: 操作指令。
            screen: 当前屏幕。

        Returns:
            执行结果。
        """
        action_type = action["type"]
        result = {"success": True, "action": action}

        if action_type == "left_click":
            target = screen.find_element_at(action["x"], action["y"])
            result["clicked"] = target or "空白区域"
            if target is None:
                result["success"] = False
                result["error"] = "未找到可点击的元素"

        elif action_type == "type_text":
            result["input"] = action["text"]

        elif action_type == "scroll":
            result["scroll"] = action.get("direction", "down")

        self.action_log.append(result)
        return result

    def run_task(self, task: str, screen: VirtualScreen) -> dict:
        """运行完整的 Computer Use 任务。

        Args:
            task: 任务描述。
            screen: 虚拟屏幕环境。

        Returns:
            包含执行记录的结果。
        """
        print(f"\n  🎯 任务: {task}")
        print(f"  📺 {screen.describe()}\n")

        for step in range(1, self.max_actions + 1):
            print(f"  --- Step {step} ---")

            # 1. 分析屏幕
            analysis = self.analyze_screen(screen)
            print(f"  👁️  分析: 发现 {analysis['elements_found']} 个元素")

            # 2. 决策
            action = self.decide_action(task, screen)
            if action is None:
                print(f"  ✅ 任务完成，无需更多操作")
                break
            print(f"  🤔 决策: {action['reasoning']}")

            # 3. 执行
            result = self.execute_action(action, screen)
            status = "✅" if result["success"] else "❌"
            print(f"  🖱️  执行: {status} {action['type']} → {result.get('clicked', result.get('input', ''))}")

            # 4. 判断是否完成
            if action.get("target") and result["success"]:
                print(f"  🎉 成功点击目标元素，任务完成！")
                break

            time.sleep(0.3)  # 模拟操作延迟

        return {
            "task": task,
            "steps": len(self.action_log),
            "actions": self.action_log,
        }


def demo_computer_use():
    """演示 Computer Use 的完整流程。"""
    print("=" * 60)
    print("  Computer Use Agent 演示")
    print("=" * 60)

    # 场景1：模拟「登录网页」
    print("\n  ── 场景1：登录网页 ──")
    login_screen = VirtualScreen(1280, 800)
    login_screen.add_element("username_input", 500, 300, 200, 30, "请输入用户名")
    login_screen.add_element("password_input", 500, 350, 200, 30, "请输入密码")
    login_screen.add_element("login_button", 550, 400, 100, 40, "登录")

    agent = ComputerUseAgent()
    result = agent.run_task("点击登录按钮", login_screen)

    # 场景2：模拟「搜索」
    print("\n  ── 场景2：搜索操作 ──")
    search_screen = VirtualScreen(1280, 800)
    search_screen.add_element("search_box", 400, 200, 400, 35, "搜索...")
    search_screen.add_element("search_button", 810, 200, 80, 35, "搜索")
    search_screen.add_element("result1", 400, 300, 500, 50, "结果1: Python教程")
    search_screen.add_element("result2", 400, 360, 500, 50, "结果2: AI Agent 入门")

    agent = ComputerUseAgent()
    result = agent.run_task("点击搜索按钮", search_screen)


"""
17.7 本章总结
━━━━━━━━━━━━━

核心要点回顾：

1. Computer Use = AI 用人类的方式操作电脑
   - Screenshot-Action Loop: 截图→分析→决策→执行
   - 不依赖 API，可以操作任何软件
   - 核心挑战：像素坐标计算 + 视觉理解

2. 两大阵营
   - Anthropic: 控制真实电脑（通用但危险）
   - OpenAI CUA: 虚拟浏览器（安全但局限）

3. 当前局限性（面试时坦诚讨论）
   - OSWorld 14.9%（人类 75%）—— 还有很长的路
   - 延迟 3-4秒/步，成本 100倍于 API
   - 安全风险高

4. 安全第一
   - Docker 沙箱 + 非管理员 + 只读挂载
   - 操作确认 + 审计日志
   - 「Always sandbox. Never admin.」

面试速记：
  "Computer Use 的原理和挑战？"
  → 原理：Screenshot-Action Loop（截图→视觉分析→坐标→操作）
  → 挑战：像素坐标精确度、延迟、安全风险
  → 定位：不是替代传统自动化，而是覆盖「API无法触及」的长尾
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第17章：Computer Use + GUI Agent                     ║")
    print("║  Screenshot-Action Loop · 坐标计算 · 安全沙箱        ║")
    print("╚══════════════════════════════════════════════════════╝")

    demo_computer_use()

    print("\n▶ OSWorld Benchmark 成绩")
    print("-" * 50)
    print("  人类                75.0%")
    print("  Claude 3.5 Sonnet  14.9%  (Computer Use)")
    print("  GPT-4V              7.8%  (传统视觉模式)")
    print()
    print("  结论：Computer Use 翻倍了前最好成绩，")
    print("        但离人类水平仍有巨大差距。")

    print("\n▶ Anthropic vs OpenAI Computer Use 对比")
    print("-" * 50)
    comparisons = [
        ("操作范围", "Anthropic: 整个操作系统", "OpenAI: 浏览器内"),
        ("安全性", "Anthropic: 需自行沙箱", "OpenAI: 内置安全隔离"),
        ("适用场景", "Anthropic: 桌面软件+Web", "OpenAI: Web任务"),
        ("成本", "Anthropic: 截图Token较贵", "OpenAI: 操作Token较低"),
    ]
    for dim, a, o in comparisons:
        print(f"  {dim:10s}  {a:30s}  {o}")

    print("\n✅ 第17章完成！")
