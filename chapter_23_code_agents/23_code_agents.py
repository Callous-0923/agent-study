"""
第23章：代码 Agent 架构横评 —— 谁是最好的 AI 程序员？
========================================================

📌 本章目标：
  1. 理解代码 Agent 的 4 种核心架构模式
  2. 掌握主要代码 Agent 的架构差异和选型策略
  3. 理解 SWE-bench 评测体系
  4. 学会「Agentless vs Agent」的设计哲学辩论

📌 面试高频点：
  - 「代码 Agent 有哪些架构类型？」
  - 「Claude Code 和 SWE-Agent 的区别？」
  - 「Agentless 为什么能超过 Agent？」
  - 「SWE-bench 是怎么评测的？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基于 2025-2026 年最新代码 Agent 生态全景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


23.1 代码 Agent 生态全景图
━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────────────────────────────────────┐
  │              代码 Agent 生态 (2025-2026)              │
  │                                                       │
  │  第一梯队（商业/闭源）     第二梯队（开源）             │
  │  ┌──────────────────┐  ┌──────────────────┐         │
  │  │ Claude Code      │  │ OpenHands        │         │
  │  │ (Anthropic)      │  │ (原 OpenDevin)   │         │
  │  │ 闭源·生产级       │  │ 开源·企业级       │         │
  │  ├──────────────────┤  ├──────────────────┤         │
  │  │ Devin            │  │ SWE-Agent        │         │
  │  │ (Cognition)      │  │ (Princeton)      │         │
  │  │ 闭源·端到端       │  │ 开源·研究导向     │         │
  │  ├──────────────────┤  ├──────────────────┤         │
  │  │ Cursor Agent     │  │ Aider            │         │
  │  │ (Anysphere)      │  │ 开源·终端优先     │         │
  │  │ 闭源·IDE集成      │  ├──────────────────┤         │
  │  ├──────────────────┤  │ Cline/Roo Code   │         │
  │  │ GitHub Copilot   │  │ 开源·VS Code插件  │         │
  │  │ Agent Mode       │  └──────────────────┘         │
  │  │ (Microsoft)      │                                │
  │  └──────────────────┘                                │
  └──────────────────────────────────────────────────────┘


23.2 四种核心架构模式
━━━━━━━━━━━━━━━━━━━━

┌─────────────────┬─────────────────────┬──────────────────────┐
│    架构          │      代表            │       核心思想         │
├─────────────────┼─────────────────────┼──────────────────────┤
│ Code-as-Action  │ OpenHands (CodeAct) │ 用代码本身作为工具接口   │
│ (CodeAct)       │                     │ 不预定义工具，写脚本执行  │
├─────────────────┼─────────────────────┼──────────────────────┤
│ Agent-Computer  │ SWE-Agent           │ 为 LLM 设计专用交互界面  │
│ Interface (ACI) │                     │ 像 HCI，但针对 AI 优化   │
├─────────────────┼─────────────────────┼──────────────────────┤
│ Plan-and-Execute│ Devin, Plandex      │ 先制定计划，人工审核后执行│
│                 │                     │ 安全优先，可审计         │
├─────────────────┼─────────────────────┼──────────────────────┤
│ React-and-Iterate│ Claude Code, Aider │ 观察→推理→行动→观察    │
│                 │ Cline, Roo Code     │ 最像人类的开发流程      │
└─────────────────┴─────────────────────┴──────────────────────┘

1. Code-as-Action (CodeAct)

  OpenHands 的创新：
    不预定义 read_file/write_file 等固定工具
    Agent 直接写 Python/Bash 脚本来完成任务

  对比传统 tool-calling：
    传统: tools = [read_file, grep, edit, ...]
    CodeAct: Agent 写 `for f in glob("*.py"): ...` 来操作

  优势：无限可扩展（任何脚本能做的事 Agent 都能做）
  劣势：可靠性低（执行任意代码的失败面更大）

2. Agent-Computer Interface (ACI)

  SWE-Agent 的创新（Princeton）：
    像设计 HCI（人机交互界面）一样设计 AI 的交互界面
    文件查看看带行号、编辑器按行号操作...

  关键设计：
    - 文件查看器显示行号（方便 LLM 定位）
    - 编辑器按行号范围操作（不是原始文本流）
    - 搜索结果带上下文
    - 命令执行结果被结构化

3. Plan-and-Execute

  代表：Devin
  安全优先：先规划 → 人工审核 → 沙箱执行
  适合企业级场景（不能允许 Agent 自己删文件）

4. React-and-Iterate（也即标准 ReAct）

  代表：Claude Code（Ch8 深入讲过）
  像程序员一样工作：搜索 → 读代码 → 改 → 测试 → 修


23.3 SWE-bench 评测体系
━━━━━━━━━━━━━━━━━━━━━━

SWE-bench 是代码 Agent 的「ImageNet」。

评测任务：
  从 GitHub 真实 Issue 中提取 → 让 Agent 修复 → 运行测试验证

两个版本：

  SWE-bench Full (2294 个任务)：
    所有真实 issue，但部分 issue 描述不清晰

  SWE-bench Verified (500 个任务)：
    人工精选，确保问题描述准确、可独立复现

  SWE-bench Lite (300 个任务)：
    进一步精简，适合快速评测

评测流程：
  1. 给 Agent GitHub Issue 描述
  2. Agent 需要在代码库中找到问题 → 修复
  3. 生成 patch → 运行原始测试 → 看是否通过

SWE-bench Verified 成绩对比 (2025 数据)：

  ┌──────────────────────┬───────────┬──────────┐
  │ Agent                 │ 解决率     │  成本     │
  ├──────────────────────┼───────────┼──────────┤
  │ Claude Code (Sonnet 4)│  ~72%     │  ~$5/任务 │
  │ OpenHands             │  ~72%     │  ~$3/任务 │
  │ SWE-Agent             │  ~45%     │  ~$1/任务 │
  │ Agentless (无 Agent!)  │  ~32%     │  $0.70    │
  │ Devin                 │  未公开    │  订阅制    │
  └──────────────────────┴───────────┴──────────┘


23.4 Agentless 的颠覆性发现
━━━━━━━━━━━━━━━━━━━━━━━━━━

2025年 Illinois 的论文：Agentless

核心发现：
  简单固定流程 = 定位 → 修复 → 验证
  不让 LLM 做任何自主决策！

  Agentless 的三阶段流程：
    1. Localization（定位）：LLM 扫描文件，找出需要修改的位置
    2. Repair（修复）：LLM 生成 patch
    3. Validation（验证）：运行测试

  和 Agent 的关键区别：
    ✗ 没有循环
    ✗ 没有工具调用
    ✗ 没有自主决策
    → 居然比很多 Agent 效果更好！

  为什么？

  1. LLM 还不够「聪明」到能做好的自主决策
  2. Agent 循环增加了出错机会（多做多错）
  3. Agent 消耗了 token 在「思考过程」而非「解决问题」上

  Anthropic 的回应（在 Claude Code 的优化中）：
    「做最简单的事」
    搜索用 Grep 而不是 Embedding
    记忆用 Markdown 而不是数据库
    单线程而不是多 Agent 并行

  这条哲学和 Agentless 的发现是一致的。


23.5 代码 Agent 的通用工作流
━━━━━━━━━━━━━━━━━━━━━━━━━━

不管什么架构，代码 Agent 的执行流程都可以抽象为：

  ┌──────────────────────────────────────────┐
  │                                             │
  │  1. 理解任务 (Understand)                   │
  │     → 读取 Issue / PR 描述                   │
  │     → 分析项目结构                           │
  │                                             │
  │  2. 定位问题 (Localize)                     │
  │     → Grep 搜索相关代码                      │
  │     → 阅读相关文件                           │
  │     → 理解调用关系                           │
  │                                             │
  │  3. 制定方案 (Plan)                         │
  │     → 确定修改哪些文件                       │
  │     → 确定怎么改                             │
  │                                             │
  │  4. 实施修改 (Implement)                    │
  │     → 编辑文件                               │
  │                                             │
  │  5. 验证 (Verify)                           │
  │     → 运行测试                               │
  │     → 如果失败 → 回到第 2 步                 │
  │                                             │
  └──────────────────────────────────────────┘
"""

import time


class CodeAgentSimulator:
    """模拟代码 Agent 的通用工作流。

    展示任意代码 Agent 必须经历的 5 个阶段。
    """

    def __init__(self, name: str, architecture: str):
        self.name = name
        self.architecture = architecture
        self.steps_log = []

    def understand(self, task: str) -> dict:
        """阶段 1：理解任务。"""
        self._log("understand", f"分析任务: {task[:60]}...")
        return {"task_type": "bugfix", "scope": "medium"}

    def localize(self, task: str, understanding: dict) -> list[str]:
        """阶段 2：定位问题。"""
        self._log("localize", "搜索相关的文件和代码")
        # 模拟：基于关键词猜测文件位置
        keywords = task.lower().split()
        files = [f"src/{w}.py" for w in keywords if len(w) > 3][:2]
        self._log("localize_result", f"定位到: {files}")
        return files or ["src/main.py"]

    def plan(self, task: str, files: list[str]) -> list[dict]:
        """阶段 3：制定方案。"""
        self._log("plan", f"针对 {len(files)} 个文件制定修改方案")
        return [
            {"file": f, "action": "edit", "lines": "45-60",
             "reason": "函数逻辑需要修改"}
            for f in files
        ]

    def implement(self, plan: list[dict]) -> list[str]:
        """阶段 4：实施修改。"""
        patches = []
        for step in plan:
            self._log("edit", f"修改 {step['file']} L{step['lines']}")
            patches.append(f"--- {step['file']}\n+++ {step['file']}\n@@ -50,10 +50,10 @@")
        return patches

    def verify(self, patches: list[str],
               task: str = "") -> tuple[bool, str]:
        """阶段 5：验证。"""
        self._log("test", "运行测试...")
        # 模拟：70% 概率通过
        passed = hash(task) % 10 < 7
        if passed:
            self._log("verify_pass", "所有测试通过")
        else:
            self._log("verify_fail", "测试失败，需要修改")
        return passed, "3/3 tests passed" if passed else "1/3 tests failed"

    def run(self, task: str, max_iterations: int = 3) -> dict:
        """运行完整的代码修复流程。"""
        print(f"\n  🤖 {self.name} ({self.architecture})")
        print(f"  📋 任务: {task[:80]}...")

        # 阶段 1-3：准备
        understanding = self.understand(task)
        files = self.localize(task, understanding)

        for iteration in range(1, max_iterations + 1):
            print(f"\n  ── 第 {iteration} 轮 ──")
            plan = self.plan(task, files)
            patches = self.implement(plan)
            passed, message = self.verify(patches, task)

            if passed:
                print(f"  ✅ 修复成功！（{iteration} 轮）")
                return {
                    "success": True,
                    "iterations": iteration,
                    "patches": patches,
                    "steps": len(self.steps_log),
                }
            else:
                print(f"  ❌ {message}，继续修改...")
                files = self.localize(task + " " + message, understanding)

        return {"success": False, "iterations": max_iterations}

    def _log(self, action: str, detail: str):
        self.steps_log.append({
            "action": action,
            "detail": detail,
            "time": time.time(),
        })
        print(f"    [{action:15s}] {detail}")


def demo_code_agents():
    """演示代码 Agent 的通用工作流。"""
    print("=" * 60)
    print("  代码 Agent 通用工作流演示")
    print("=" * 60)

    # 测试不同架构的 Agent
    agents = [
        CodeAgentSimulator("ClaudeCode", "React-and-Iterate"),
        CodeAgentSimulator("OpenHands", "Code-as-Action"),
        CodeAgentSimulator("SWE-Agent", "Agent-Computer Interface"),
    ]

    task = "修复 login.py 中的空指针异常：当 user 为 None 时访问 user.name 会崩溃"

    for agent in agents:
        result = agent.run(task)
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"  {status} | 迭代: {result['iterations']} | 步骤: {result['steps']}")


"""
23.6 选型指南
━━━━━━━━━━━━

  ┌───────────────────┬────────────────────────────────────┐
  │       场景         │              推荐                    │
  ├───────────────────┼────────────────────────────────────┤
  │ 个人开发者日常     │ Claude Code / Aider                 │
  │ 企业级代码审查     │ OpenHands (企业版)                  │
  │ 安全关键系统       │ Plan-and-Execute (人工审核)         │
  │ 研究和实验         │ SWE-Agent (ACI 研究)               │
  │ 定制化 CI/CD      │ 自建 (参考 Agentless 简洁方案)       │
  │ IDE 深度集成      │ Cursor Agent / GitHub Copilot Agent  │
  │ 多文件重构         │ Claude Code / OpenHands             │
  └───────────────────┴────────────────────────────────────┘

23.7 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. 4 种架构模式
   CodeAct / ACI / Plan-Execute / React-and-Iterate

2. SWE-bench 是代码 Agent 的标准评测
   Verified 版最可靠，Lite 版适合快速验证

3. Agentless 的启示
   简单方案常比复杂 Agent 更好
   验证了 Anthropic「做最简单的事」的哲学

面试速记：
  "你了解哪些代码 Agent？"
  → 4 种架构 + 每个代表 + 选型场景
  → 提到 SWE-bench 评测体系
  → Agentless 的「返璞归真」启示
  → Claude Code 的简单至上哲学
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第23章：代码 Agent 架构横评                            ║")
    print("║  CodeAct · ACI · Plan-Execute · SWE-bench           ║")
    print("╚══════════════════════════════════════════════════════╝")

    demo_code_agents()

    print("\n▶ 四种架构模式对比")
    print("-" * 50)
    archs = [
        ("CodeAct (OpenHands)", "用代码脚本替代预定义工具"),
        ("ACI (SWE-Agent)", "为 LLM 设计专用交互界面"),
        ("Plan-Execute (Devin)", "先规划→审核→沙箱执行"),
        ("React-Iterate (Claude Code)", "像人类一样渐进式解决问题"),
    ]
    for name, desc in archs:
        print(f"  {name:25s} → {desc}")

    print("\n▶ SWE-bench Verified 成绩")
    print("-" * 50)
    print("  Claude Code (Sonnet 4)   ~72%     ~$5/task")
    print("  OpenHands                 ~72%     ~$3/task")
    print("  SWE-Agent                 ~45%     ~$1/task")
    print("  Agentless (无Agent!)      ~32%     $0.70")

    print("\n✅ 第23章完成！")
