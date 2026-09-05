# 2026-08 Agent 全栈内容更新报告

> 状态：已完成。内容核对基线为 2026-08-01，本轮修复完成于 2026-08-09。

## Scope

- Ch0-Ch36 Python 讲义与演示
- 生成的 HTML 章节和站点索引
- 中英文 README
- 安装、依赖、版本和验证方式

## Final Summary

Ch0-Ch36 共 37 章已经完成源码、静态页面、依赖和入口文档的一致性更新。重点不是简单替换模型名称，而是修复协议字段、真实项目事实、生产架构边界、不安全示例和缺失的前后端链路。

## Important Fixes

### 协议与当前平台能力

- Ch10 按 MCP 2026-07-28 补充无状态路由头、`server/discover`、`ttlMs`、扩展与任务边界。
- Ch15 将 A2A v1 事件包装字段修正为 `statusUpdate` 与 `artifactUpdate`，移除旧操作和错误字段。
- Ch11 加入 OpenAI Tool Search、Programmatic Tool Calling、`allowed_callers`、`output_schema`、`previous_response_id` 与程序输出循环。
- Ch13 加入 Background Mode、Webhook 验签、幂等和队列消费边界。
- Ch21 加入前端事件契约、取消/恢复、审批事件，以及 Realtime Voice 的 WebRTC、WebSocket、SIP 与打断处理。

### Agent 全栈与工程实现

- Ch5 补充 LangGraph、OpenAI Agents SDK、Responses 多 Agent 编排和自定义工作流引擎的选型边界。
- Ch13 现在包含可以直接打开的浏览器控制台，通过 `fetch` 消费 POST SSE，并支持取消请求；后端保留健康检查和流式接口。
- Ch12 修正 OpenClaw 与 OpenHarness 的项目事实：前者是自托管 Agent Gateway，后者是 code-first TypeScript Agent SDK；教学评测 Harness 被明确标注为课程示意而非真实产品。
- Ch4 与 Ch11 的计算工具移除 `eval`，改为 AST 白名单算术解析，拒绝名称、属性和函数调用。
- Ch29 将视觉示例改为配置化的当前视觉模型，不再把 GPT-4o Vision 或 Claude 3.5 当作当前推荐。

### 内容可信度与可维护性

- Ch7、Ch9、Ch17、Ch25 删除或改写了无来源的岗位、薪资、延迟、质量提升、成本倍数、失败率和固定阈值。
- 所有教学模拟数值都与真实 SDK、协议结果或公开基准区分；生产参数改为由数据集、流量和评测确定。
- README、英文 README、站点索引和 `package.json` 统一为 Ch0-Ch36 共 37 章，并更新当前技术栈说明。
- Python 依赖增加兼容 Python 3.10 的 NumPy 条件分支，并提供 `openai-agents` 可选依赖。

## Generated Site Integrity

- `python build_html.py --all` 会重建全部 37 个章节页面。
- 每个生成页面嵌入源文件 SHA-256；测试会检测 HTML 是否落后于 Python 源文件。
- 修复了嵌套引号导致的语法高亮占位符泄漏，并固定 LF 换行、清理尾随空白。

## Verification

- `python -m pytest -q -p no:cacheprovider`：9 项测试全部通过。
- 37 个 Python 章节及生成器均通过 AST 语法解析。
- FastAPI 演示通过浏览器首页、健康检查和 POST SSE 集成测试。
- Python 3.10.10 下，基础依赖及 `openai-agents>=0.6,<1` 均通过 `pip --dry-run` 解析。
- `git diff --check` 通过；已知旧协议字段、旧模型推荐、错误项目描述和不安全 `eval` 的回归扫描通过。

## Primary Sources

- OpenAI model guidance: https://developers.openai.com/api/docs/guides/latest-model
- OpenAI Tool Search: https://developers.openai.com/api/docs/guides/tools-tool-search
- OpenAI Programmatic Tool Calling: https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling
- OpenAI Background Mode: https://developers.openai.com/api/docs/guides/background
- OpenAI Webhooks: https://developers.openai.com/api/docs/guides/webhooks
- MCP 2026-07-28 change summary: https://blog.modelcontextprotocol.io/posts/2026-07-28/
- A2A v1 changes: https://a2a-protocol.org/latest/whats-new-v1/
- LangChain v1 release notes: https://docs.langchain.com/oss/python/releases/langchain-v1
- OpenClaw documentation: https://docs.openclaw.ai/
- OpenHarness quickstart: https://docs.open-harness.dev/getting-started/quickstart
