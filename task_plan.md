# Task Plan: AI Agent 课程全量审计与更新

## Goal
将 Ch0-Ch36、README、依赖说明和静态站点更新到 2026-08-09 可验证的官方规范与主流 API，并完成一致性、语法、运行和生成物检查。

## Phases
- [x] Phase 1: 盘点仓库、保护用户改动并确定审计范围
- [x] Phase 2: 核对官方规范、API、模型、框架与评测资料
- [x] Phase 3: 更新课程元数据、依赖和 Ch0-Ch18
- [x] Phase 4: 更新 Ch19-Ch36、README 和英文 README
- [x] Phase 5: 重建 HTML、执行静态与运行验证
- [x] Phase 6: 完成逐章审计报告并交付

## Key Questions
1. 哪些内容属于稳定方法论，哪些必须随 2026 年规范更新？
2. 示例应如何区分教学模拟、真实 SDK 示例和生产建议？
3. 如何避免模型价格、排行榜和产品状态再次快速失效？
4. 如何在不覆盖现有用户改动的前提下重建静态站点？

## Decisions Made
- 以 2026-08-01 为全课程内容核对基线，以 2026-08-09 为本轮修复日期；动态事实使用官方资料并标注日期。
- 保留 Ch0-Ch36 的课程结构，统一称为 37 章。
- 保留教学模拟器，但必须显式标注其不等于真实 SDK/协议实现。
- OpenAI 新示例优先 Responses API；保留 Chat Completions 仅用于对比或兼容说明。
- HTML 统一由同名 Python 源文件生成；每页嵌入源文件 SHA-256，测试阻止过期生成物混入站点。

## Errors Encountered
- 初次记忆检索无项目相关记录：停止使用记忆，改以仓库和官方资料为准。
- PowerShell 初次读取 UTF-8 README 出现乱码：后续显式使用 UTF-8 或 `rg`。
- OpenAI 最新模型解析器两次请求 `latest-model.md` 均返回 HTTP 403：按技能规则切换到官方 Docs MCP；若不可用则仅使用 OpenAI 官方域名网页。
- HTML 高亮器最初会在嵌套引号中残留内部占位符：改为逆序还原并加入失败保护和回归测试。
- Windows 生成物最初产生换行及尾随空白差异：生成器固定 LF 并清理行尾空白。

## Status
**Complete** - 37 章源文件与 HTML、README、索引、依赖和自动化验证均已完成更新。
