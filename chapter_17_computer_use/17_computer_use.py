"""
第17章：Computer Use + GUI Agent —— AI 操控电脑
==================================================

内容核对：2026-08-01
说明：标注为模拟的实现与数值用于讲解概念，不代表真实 SDK、协议或基准结果。

📌 本章目标：
  1. 理解 Computer Use / GUI Agent 的核心原理（Screenshot-Action Loop）
  2. 掌握 Anthropic 与 OpenAI Computer Use 工具的共同循环与 API 差异
  3. 理解像素坐标的计算与视觉定位机制
  4. 了解 Browser Use 等开源方案
  5. 认识安全沙箱的必要性和实现方式

📌 面试高频点：
  - Computer Use 的原理是什么？和传统 API 调用有什么区别？
  - Screenshot-Action Loop 的每一步做了什么？
  - OpenAI 与 Anthropic Computer Use 的 API 与责任边界有何差异？
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

  模型和运行时需要共同解决视觉定位：模型提出动作，执行器将坐标映射到
  当前截图/视口，并在执行后重新截图验证状态。

  实操中的坐标系统：
    - 截图尺寸通常是 1280x800 或 1920x1080
    - LLM 返回的坐标需要缩放到实际屏幕分辨率
    - 返回格式：(x_pct, y_pct) 百分比比绝对像素更稳健


17.3 Anthropic vs OpenAI Computer Use
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────┬─────────────────────────┬─────────────────────────┐
│     维度      │  Anthropic Computer Use  │   OpenAI Computer Use   │
├──────────────┼─────────────────────────┼─────────────────────────┤
│ 产品状态      │ Beta，需使用 beta header │ Responses 中的 computer  │
│ 交互模型      │ 返回 computer tool actions│ 返回 computer actions    │
│ 执行环境      │ 开发者提供并负责隔离       │ 开发者/平台提供受控环境    │
│ 动作类型      │ 鼠标、键盘、截图等         │ 点击、输入、滚动等        │
│ 反馈循环      │ 执行动作后回传新截图       │ 执行动作后回传环境输出     │
│ 风险控制      │ allowlist + 人工确认       │ allowlist + 人工确认      │
└──────────────┴─────────────────────────┴─────────────────────────┘

Anthropic 的设计哲学：
  「给 Claude 真实的电脑，让它按人类的方式工作」

OpenAI 当前新项目使用 Responses API 的 computer 工具；旧的
`computer-use-preview` 模型/工具路径已进入弃用路线，迁移时应查官方指南。

选型建议：
  - 需要控制桌面软件 → Anthropic Computer Use
  - 浏览器操作 → 两者都需在受控浏览器中做任务集评测
  - 桌面操作 → 选择支持目标环境的工具，并由开发者提供强隔离


17.4 性能评测与局限
━━━━━━━━━━━━━━━━━━━

公开 benchmark 分数会随模型、截图分辨率、运行环境和 harness 快速变化。
上线前至少记录：任务成功率、危险动作拦截率、平均/尾部动作数、p95 延迟、
输入/输出 token、人工接管率和恢复时间；报告必须带模型与评测日期。

当前定位（截至 2026-08-01）：
  → 不是 Selenium 的替代品
  → 适合「API 无法覆盖的长尾场景」
  → 适合「快速原型验证」
  → 两家工具都要求开发者承担沙箱、确认和结果验证责任


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


17.5.1 坐标系统工程 —— 为什么 LLM 总是点不准？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▍ 坐标转换：从 LLM 输出到屏幕像素

  LLM 输出的坐标通常是「归一化坐标」或「基于截图分辨率的坐标」。
  但实际屏幕分辨率可能不同 → 需要坐标缩放。

  问题链：
    截图 1280x800 → LLM 分析 → 输出「点击 (640, 400)」→ 但实际屏幕是 2560x1600
    → 如果直接发 (640, 400)，点到错误位置！

  标准做法：
    a) 发给 LLM 的截图使用固定分辨率（如 1280x800）
    b) LLM 输出坐标基于 1280x800
    c) 执行前做坐标缩放：x_real = x_llm × (screen_width / screenshot_width)
    d) 使用百分比坐标更稳健：(x_pct, y_pct) 而非绝对像素

▍ 为什么 LLM 的坐标会有 ±5px 的误差？

  这是 Computer Use 的一个核心难点：「让文本模型理解空间坐标」。

  Anthropic 专门训练了 Claude 的「像素计数能力」，但仍然存在天然误差：
    - 小按钮（20x20px）→ 5px 误差可能点到按钮外
    - 密集列表 → 可能点到相邻项
    - 动态 UI（动画中的元素）→ 坐标过了检测时已经变了

  工程应对：
    a) 坐标容错 —— 点击后截图验证，不符预期则微调坐标重试
    b) 区域点击代替点点击 —— 「点击按钮中心区域」而非精确坐标
    c) 元素描述作为主导航 —— 「点击 'Login' 按钮」→ 先用 OCR 定位

▍ 截图分辨率的经济学

  分辨率越高 → LLM 的坐标越精准 → 但 Token 成本越高
  分辨率越低 → 成本低 → 但可能 LLM 看错按钮

  分辨率策略不能套用跨模型固定 token 表：
    - 先从满足可读性的较低分辨率开始
    - OCR、代码或表格识别失败时，再提高分辨率或裁剪重点区域
    - 从响应 usage 和账单记录每类截图的真实成本
    - 模型、detail、缩放规则变化后重新测量，不沿用旧估算

  动态分辨率策略：
    第一轮 → 低分辨率 1024x768（快速判断页面状态）
    发现无法识别 → 升级到 1280x800
    仍然需要细节 → 局部截图 500x500（放大特定区域）


17.5.2 沙箱深度 —— 不是「run docker」就完了

▍ 多层安全隔离（从外到内 4 层）

  第1层：容器隔离 —— Docker/VM 层
    禁用网络（--network=none）、只读根文件系统（--read-only）、
    限制 CPU/Memory、丢弃所有 Linux Capabilities

  第2层：用户隔离 —— 容器内部跑非 root 用户
    USER agent（非 root），家目录只读
    授予的目录要白名单管理，不是黑名单

  第3层：动作白名单 —— 不是所有键盘组合都允许
    禁止: Ctrl+Alt+Del、Win+R、rm -rf、格式化命令
    允许: 文本输入、鼠标移动、窗口切换

  第4层：结果审查 —— Agent 操作完的截图交给另一个 LLM 审核
    检查：是否打开了敏感页面？是否输入了敏感内容？
    这是最后一道防线

▍ 成本优化 —— Computer Use 的「省钱三板斧」

  1. 缓存重复截图 —— 同一个页面多次截取 → 对比 hash，相同则复用 LLM 分析结果
  2. 最小化截图区域 —— 不全屏截图，只截需要操作的应用窗口
  3. 混合自动化 —— 能用稳定 API 或 DOM 自动化的操作优先走确定性路径，
     只在这些路径无法覆盖时启动 Computer Use；成本差异必须按本系统实测

  面试可以提：「我们不是用 Computer Use 替代 Selenium，而是把它用于
  API/DOM 自动化无法覆盖的长尾场景，并单独测量成功率、成本与人工接管率。」


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
   - Anthropic: Beta computer tool，运行环境由开发者负责
   - OpenAI: Responses computer tool；旧 preview 路径需迁移

3. 当前局限性（面试时坦诚讨论）
   - benchmark 分数依模型、环境和 harness 变化，必须带日期复测
   - 多轮截图-动作会累积延迟、token 和失败概率
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

    print("\n▶ Computer Use 评测维度")
    print("-" * 50)
    print("  成功率 / 危险动作拦截率 / 人工接管率")
    print("  动作数 / p95 延迟 / token / 恢复时间")
    print("  所有结果必须附模型、环境、harness 和日期")

    print("\n▶ Anthropic vs OpenAI Computer Use 对比")
    print("-" * 50)
    comparisons = [
        ("产品状态", "Anthropic: Beta", "OpenAI: Responses tool"),
        ("执行环境", "开发者提供隔离环境", "开发者/平台受控环境"),
        ("反馈循环", "动作后回传截图", "动作后回传环境输出"),
        ("安全责任", "allowlist + 人工确认", "allowlist + 人工确认"),
    ]
    for dim, a, o in comparisons:
        print(f"  {dim:10s}  {a:30s}  {o}")

    print("\n✅ 第17章完成！")
