"""
第14章：SQLite + Agent 持久化存储
==================================

📌 本章目标：
  1. 用 SQLite 实现 Agent 的会话持久化（对话历史不丢失）
  2. 实现任务状态管理（暂停/恢复/取消）
  3. 实现用户管理系统（配额/权限/偏好）
  4. 掌握 Agent 场景下的 SQL 设计模式
  5. 实现分析查询（Token 消耗 / 工具使用频率 / 成功率）
  6. 学习 WAL 模式、连接池、并发安全等生产实践

📌 面试高频点：
  - Agent 的数据库 schema 怎么设计？
  - 对话历史存在哪里？怎么高效查询？
  - SQLite 的 WAL 模式是什么？为什么 Agent 场景需要？
  - 如何实现 Agent 任务的暂停和恢复？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本章完全独立运行（仅依赖 Python 标准库 sqlite3）
运行：python chapter_14_sqlite/14_sqlite_agent_storage.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


14.1 为什么是 SQLite？
━━━━━━━━━━━━━━━━━━━━━━━

❌ 常见误区：「Agent 项目 = 必须用 PostgreSQL / MongoDB」

✅ 真相：SQLite 是 Agent 开发中最被低估的数据库。

SQLite 在 Agent 场景中的优势：
  1. 零配置：不需要安装数据库服务，文件即数据库
  2. 嵌入式：数据库和应用在同一个进程（低延迟）
  3. 便携性：一个 .db 文件可以备份、迁移、版本控制
  4. WAL 模式：支持并发读 + 单写（Agent 场景足够）
  5. 全文搜索(FTS5)：内置全文索引（对话搜索）
  6. JSON 支持：可以存储半结构化数据

什么时候用 SQLite？什么时候升级到 PostgreSQL？

  SQLite 适用：
    ✓ 单机部署的 Agent 服务
    ✓ 原型开发 / MVP 阶段
    ✓ 个人 Agent 助手
    ✓ 中小型团队内部工具
    ✓ 对话历史 < 10M 条

  PostgreSQL 适用：
    → 多副本高可用需求
    → 写入 QPS > 1000
    → 需要行级权限控制
    → 需要地理分布部署


14.2 Agent 数据库 schema 设计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心表设计（5 张表）：

  ┌────────────────────────────────────────────────────┐
  │                    SQLite 数据库                     │
  │                                                      │
  │  ┌─────────────┐  ┌──────────────────┐             │
  │  │   sessions   │  │    messages       │             │
  │  │  ─────────  │  │   ────────────   │             │
  │  │  id (PK)    │  │   id (PK)         │             │
  │  │  user_id    │──│   session_id (FK) │             │
  │  │  title      │  │   role            │             │
  │  │  status     │  │   content         │             │
  │  │  created_at │  │   tool_calls_json │             │
  │  │  updated_at │  │   token_count     │             │
  │  └─────────────┘  │   created_at      │             │
  │                    └──────────────────┘             │
  │                                                      │
  │  ┌─────────────┐  ┌──────────────────┐             │
  │  │    tasks     │  │    tool_logs      │             │
  │  │  ─────────  │  │   ────────────   │             │
  │  │  id (PK)    │  │   id (PK)         │             │
  │  │  session_id │  │   session_id      │             │
  │  │  type       │  │   tool_name       │             │
  │  │  payload    │  │   input_json      │             │
  │  │  status     │  │   output_json     │             │
  │  │  result     │  │   elapsed_ms      │             │
  │  │  created_at │  │   success         │             │
  │  └─────────────┘  │   created_at      │             │
  │                    └──────────────────┘             │
  │                                                      │
  │  ┌─────────────┐                                    │
  │  │    users     │  ← 扩展用（配额/权限/偏好）        │
  │  │  ─────────  │                                    │
  │  │  id (PK)    │                                    │
  │  │  name       │                                    │
  │  │  quota_total│                                    │
  │  │  quota_used │                                    │
  │  └─────────────┘                                    │
  └────────────────────────────────────────────────────┘

设计要点（面试重点！）：
  1. sessions 和 messages 是 1:N 关系
  2. tool_calls 存储在 messages 表的 JSON 字段中（不用单独的表）
     → 减少 JOIN，提高读取性能
  3. tool_logs 是独立的审计表（记录每次工具调用的详细信息）
  4. tasks 表支持异步任务（暂停/恢复/取消）
"""

import os
import json
import time
import hashlib
import sqlite3
from datetime import datetime
from contextlib import contextmanager
from typing import Optional


DB_PATH = os.path.join(os.path.dirname(__file__), "agent_store.db")


@contextmanager
def get_db():
    """获取数据库连接（上下文管理器，自动提交/回滚）。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 结果以字典形式返回
    conn.execute("PRAGMA journal_mode=WAL")  # 启用 WAL 模式
    conn.execute("PRAGMA foreign_keys=ON")   # 启用外键约束
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """初始化数据库 —— 创建所有表。

    这是 Agent 持久化的第一步。
    每次启动时调用，用 IF NOT EXISTS 保证幂等。
    """
    with get_db() as conn:
        conn.executescript("""
        -- ===== 用户表 =====
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            api_key_hash TEXT,
            quota_total INTEGER DEFAULT 100000,
            quota_used INTEGER DEFAULT 0,
            preferences_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- ===== 会话表 =====
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT DEFAULT '新对话',
            status TEXT DEFAULT 'active'
                CHECK(status IN ('active','paused','completed','cancelled')),
            model TEXT DEFAULT 'gpt-4o-mini',
            total_tokens INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0.0,
            message_count INTEGER DEFAULT 0,
            metadata_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        -- ===== 消息表（核心！）=====
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('system','user','assistant','tool')),
            content TEXT NOT NULL DEFAULT '',
            tool_calls_json TEXT,
            tool_call_id TEXT,
            token_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        -- ===== 任务表（异步任务管理）=====
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            status TEXT DEFAULT 'pending'
                CHECK(status IN ('pending','running','paused','completed','failed','cancelled')),
            result_json TEXT,
            priority INTEGER DEFAULT 0,
            max_retries INTEGER DEFAULT 3,
            retry_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            started_at TEXT,
            completed_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        -- ===== 工具调用日志表（审计 + 分析）=====
        CREATE TABLE IF NOT EXISTS tool_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_id INTEGER,
            tool_name TEXT NOT NULL,
            input_json TEXT NOT NULL,
            output_json TEXT,
            elapsed_ms REAL,
            success INTEGER DEFAULT 1,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (message_id) REFERENCES messages(id)
        );

        -- ===== 索引（查询性能关键！）=====
        CREATE INDEX IF NOT EXISTS idx_sessions_user
            ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_updated
            ON sessions(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_messages_role
            ON messages(session_id, role);
        CREATE INDEX IF NOT EXISTS idx_tasks_status
            ON tasks(status, priority DESC);
        CREATE INDEX IF NOT EXISTS idx_tool_logs_session
            ON tool_logs(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_tool_logs_name
            ON tool_logs(tool_name, created_at);
        """)
    print("  ✅ 数据库已初始化（WAL模式 + 5张表 + 6个索引）")


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
14.3 Agent 存储层 API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这里封装一个 SQLiteAgentStore 类，提供 5 张表的 CRUD 操作。
设计思路：每个 Agent 交互环节（会话、任务、消息、工具调用、用户）都有独立表。
使用 WAL 模式保证高并发下的读写不阻塞，所有写操作都有审计时间戳。
"""


class AgentStorage:
    """Agent 持久化存储 —— 封装所有数据库操作。

    设计原则：
      1. 每个方法完成一个业务操作
      2. 内部处理事务（外部无需关心）
      3. 返回 Python 原生类型（dict/list）
    """

    # ==================== 用户管理 ====================

    @staticmethod
    def create_user(name: str, email: str = "",
                    quota_total: int = 100000) -> dict:
        """创建用户。

        Args:
            name: 用户名。
            email: 邮箱（可选）。
            quota_total: Token 配额上限。

        Returns:
            创建的用户信息字典。
        """
        user_id = hashlib.md5(f"{name}-{time.time()}".encode()).hexdigest()[:16]
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (id, name, email, quota_total) VALUES (?,?,?,?)",
                (user_id, name, email, quota_total),
            )
        return {"id": user_id, "name": name, "quota_total": quota_total}

    @staticmethod
    def get_user(user_id: str) -> Optional[dict]:
        """获取用户信息。"""
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id=?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def check_quota(user_id: str, required_tokens: int = 0) -> bool:
        """检查用户 Token 配额是否充足。

        Args:
            user_id: 用户 ID。
            required_tokens: 本次需要的 Token 数。

        Returns:
            是否有足够配额。
        """
        with get_db() as conn:
            row = conn.execute(
                "SELECT quota_total - quota_used AS remaining FROM users WHERE id=?",
                (user_id,),
            ).fetchone()
        if row is None:
            return False
        return row["remaining"] >= required_tokens

    @staticmethod
    def consume_quota(user_id: str, tokens: int):
        """消耗用户配额。"""
        with get_db() as conn:
            conn.execute(
                "UPDATE users SET quota_used=quota_used+?, updated_at=datetime('now') WHERE id=?",
                (tokens, user_id),
            )

    # ==================== 会话管理 ====================

    @staticmethod
    def create_session(user_id: str, title: str = "新对话",
                       model: str = "gpt-4o-mini") -> dict:
        """创建新会话。

        Args:
            user_id: 用户 ID。
            title: 会话标题。
            model: 使用的模型。

        Returns:
            创建的会话信息。
        """
        session_id = hashlib.md5(
            f"{user_id}-{time.time()}-{os.urandom(4).hex()}".encode()
        ).hexdigest()[:16]
        with get_db() as conn:
            conn.execute(
                """INSERT INTO sessions (id, user_id, title, model)
                   VALUES (?,?,?,?)""",
                (session_id, user_id, title, model),
            )
        return {"id": session_id, "user_id": user_id, "title": title, "model": model}

    @staticmethod
    def get_session(session_id: str) -> Optional[dict]:
        """获取会话详情。"""
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def list_user_sessions(user_id: str, limit: int = 20) -> list[dict]:
        """列出用户的所有会话（按更新时间倒序）。

        Args:
            user_id: 用户 ID。
            limit: 返回数量上限。

        Returns:
            会话列表。
        """
        with get_db() as conn:
            rows = conn.execute(
                """SELECT id, title, status, model, message_count,
                          total_tokens, updated_at
                   FROM sessions WHERE user_id=?
                   ORDER BY updated_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def pause_session(session_id: str):
        """暂停会话（暂停 Agent 执行）。"""
        with get_db() as conn:
            conn.execute(
                "UPDATE sessions SET status='paused', updated_at=datetime('now') WHERE id=?",
                (session_id,),
            )

    @staticmethod
    def resume_session(session_id: str):
        """恢复会话。"""
        with get_db() as conn:
            conn.execute(
                "UPDATE sessions SET status='active', updated_at=datetime('now') WHERE id=?",
                (session_id,),
            )

    # ==================== 消息管理（核心！）====================

    @staticmethod
    def save_message(session_id: str, role: str, content: str,
                     tool_calls: Optional[list] = None,
                     tool_call_id: Optional[str] = None,
                     token_count: int = 0):
        """保存一条消息到对话历史。

        这是 Agent 存储的核心操作，每次 LLM 交互都需要调用。

        Args:
            session_id: 会话 ID。
            role: 消息角色（user/assistant/system/tool）。
            content: 消息内容。
            tool_calls: 工具调用信息（仅 assistant 消息有）。
            tool_call_id: 工具调用 ID（仅 tool 消息有）。
            token_count: Token 估算值。
        """
        tool_calls_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        with get_db() as conn:
            conn.execute(
                """INSERT INTO messages (session_id, role, content,
                   tool_calls_json, tool_call_id, token_count)
                   VALUES (?,?,?,?,?,?)""",
                (session_id, role, content, tool_calls_json,
                 tool_call_id, token_count),
            )
            # 同步更新会话统计
            conn.execute(
                """UPDATE sessions SET
                   message_count = message_count + 1,
                   total_tokens = total_tokens + ?,
                   updated_at = datetime('now')
                   WHERE id=?""",
                (token_count, session_id),
            )

    @staticmethod
    def get_conversation_history(session_id: str,
                                  max_messages: int = 50) -> list[dict]:
        """获取会话的最近 N 条消息（用于构建 LLM 上下文）。

        这是 Agent 记忆系统的核心查询。

        Args:
            session_id: 会话 ID。
            max_messages: 最大返回条数。

        Returns:
            消息列表（按时间正序）。
        """
        with get_db() as conn:
            rows = conn.execute(
                """SELECT * FROM (
                       SELECT role, content, tool_calls_json, tool_call_id,
                              token_count, created_at
                       FROM messages WHERE session_id=?
                       ORDER BY created_at DESC LIMIT ?
                   ) ORDER BY created_at ASC""",
                (session_id, max_messages),
            ).fetchall()
        return [dict(r) for r in rows]

    # ==================== 工具调用日志 ====================

    @staticmethod
    def log_tool_call(session_id: str, tool_name: str,
                      input_data: dict, output_data: dict,
                      elapsed_ms: float, success: bool = True,
                      error_message: str = None,
                      message_id: int = None):
        """记录一次工具调用。

        这是 Agent 审计和优化的基础数据。

        Args:
            session_id: 会话 ID。
            tool_name: 工具名称。
            input_data: 工具输入参数。
            output_data: 工具输出结果。
            elapsed_ms: 执行耗时（毫秒）。
            success: 是否成功。
            error_message: 错误信息。
            message_id: 关联的消息 ID。
        """
        with get_db() as conn:
            conn.execute(
                """INSERT INTO tool_logs (session_id, message_id, tool_name,
                   input_json, output_json, elapsed_ms, success, error_message)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, message_id, tool_name,
                 json.dumps(input_data, ensure_ascii=False),
                 json.dumps(output_data, ensure_ascii=False),
                 elapsed_ms, int(success), error_message),
            )

    # ==================== 任务管理 ====================

    @staticmethod
    def create_task(session_id: str, task_type: str,
                    payload: dict, priority: int = 0) -> dict:
        """创建一个异步任务。

        Args:
            session_id: 关联的会话。
            task_type: 任务类型。
            payload: 任务参数。
            priority: 优先级（越大越优先）。

        Returns:
            创建的任务信息。
        """
        task_id = hashlib.md5(
            f"{session_id}-{task_type}-{time.time()}".encode()
        ).hexdigest()[:16]
        with get_db() as conn:
            conn.execute(
                """INSERT INTO tasks (id, session_id, type, payload_json, priority)
                   VALUES (?,?,?,?,?)""",
                (task_id, session_id, task_type,
                 json.dumps(payload, ensure_ascii=False), priority),
            )
        return {"id": task_id, "type": task_type, "status": "pending"}

    @staticmethod
    def update_task_status(task_id: str, status: str,
                           result: Optional[dict] = None):
        """更新任务状态。

        Args:
            task_id: 任务 ID。
            status: 新状态。
            result: 任务结果（完成时）。
        """
        updates = ["status=?"]
        params = [status]

        if status == "running":
            updates.append("started_at=datetime('now')")
        elif status in ("completed", "failed", "cancelled"):
            updates.append("completed_at=datetime('now')")

        if result is not None:
            updates.append("result_json=?")
            params.append(json.dumps(result, ensure_ascii=False))

        params.append(task_id)

        with get_db() as conn:
            conn.execute(
                f"UPDATE tasks SET {', '.join(updates)} WHERE id=?",
                params,
            )

    # ==================== 分析查询 ====================

    @staticmethod
    def get_usage_stats(user_id: str, days: int = 7) -> dict:
        """获取用户的用量统计。

        Args:
            user_id: 用户 ID。
            days: 统计天数。

        Returns:
            包含各项统计的字典。
        """
        with get_db() as conn:
            # 总览
            total = conn.execute(
                """SELECT COUNT(*) as session_count,
                          SUM(total_tokens) as total_tokens,
                          SUM(message_count) as total_messages
                   FROM sessions
                   WHERE user_id=? AND updated_at > datetime('now', ?)""",
                (user_id, f"-{days} days"),
            ).fetchone()

            # 工具调用统计
            tools = conn.execute(
                """SELECT tool_name, COUNT(*) as call_count,
                          AVG(elapsed_ms) as avg_latency,
                          SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failures
                   FROM tool_logs
                   WHERE session_id IN (
                       SELECT id FROM sessions WHERE user_id=?
                   ) AND created_at > datetime('now', ?)
                   GROUP BY tool_name ORDER BY call_count DESC""",
                (user_id, f"-{days} days"),
            ).fetchall()

        return {
            "period_days": days,
            "session_count": total["session_count"],
            "total_tokens": total["total_tokens"] or 0,
            "total_messages": total["total_messages"] or 0,
            "tool_usage": [dict(r) for r in tools],
        }


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
14.4 关键设计决策详解（面试重点！）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

决策 1: 为什么 tool_calls 存在 messages 表里而不单独建表？

  方案 A（单独建表）:
    messages 表 + tool_calls 表 → JOIN 查询

  方案 B（JSON 字段）:
    messages 表的 tool_calls_json 字段

  选择方案 B 的理由：
    ✓ Agent 读取对话历史时，要一次性加载所有信息（含 tool_calls）
    ✓ 单表读取比 JOIN 快得多
    ✓ Agent 不会按 tool_name 筛选历史消息（没有这种查询需求）
    ✓ 但 tool_logs 表单独存在（用于审计和分析查询）

决策 2: WAL 模式为什么重要？

  WAL = Write-Ahead Logging（预写式日志）

  默认模式（DELETE）:
    写入时锁定整个数据库 → 读写互斥
    AgentA 在写入消息 → AgentB 无法读取历史

  WAL 模式:
    写入操作记录到 WAL 文件 → 不阻塞读取
    支持无限并发读 + 1 个写
    AgentA 写消息 + AgentB 读历史 = 同时进行 ✓

  代价：
    WAL 文件会增长，需要定期 checkpoint（SQLite 自动处理）

决策 3: 为什么用 TEXT 存时间而不是 TIMESTAMP？

  SQLite 没有 DATE/TIME 类型，TEXT 的 ISO8601 格式：
    ✓ 人类可读（便于调试）
    ✓ 排序正确（符合 ISO8601 字典序）
    ✓ 跨语言一致


14.5 SQLite 在 Agent 场景中的高级用法
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. FTS5 全文搜索 —— 搜索对话历史

  CREATE VIRTUAL TABLE messages_fts USING fts5(
      content,
      content='messages',        -- 外部内容表
      content_rowid='id'          -- 外部表的行ID
  );

  -- 搜索包含 "天气" 的消息
  SELECT * FROM messages_fts WHERE content MATCH '天气';

2. JSON 函数 —— 查询半结构化数据

  -- 查询偏好中 theme 为 dark 的用户
  SELECT * FROM users
  WHERE json_extract(preferences_json, '$.theme') = 'dark';

  -- 查询 tool_calls 中包含 search 工具的消息
  SELECT * FROM messages
  WHERE tool_calls_json LIKE '%"name":"search"%';

3. 增量备份

  -- VACUUM INTO 备份到新文件
  conn.execute("VACUUM INTO 'agent_store_backup.db'")


14.6 本章总结
━━━━━━━━━━━━━

核心要点回顾：

1. SQLite 是 Agent 开发中最实用的数据库
   - 零配置、嵌入式、便携
   - WAL 模式支持高并发读
   - FTS5 + JSON 支持高级查询

2. Schema 设计核心
   - sessions: 会话管理
   - messages: 对话历史（含 JSON 格式的 tool_calls）
   - tasks: 异步任务（暂停/恢复/取消）
   - tool_logs: 工具调用审计

3. 关键决策（面试重点！）
   - tool_calls 用 JSON 字段 vs 单独建表（选择 JSON 减少 JOIN）
   - WAL 模式保证读写不互斥
   - TEXT 存时间（人类可读 + 排序正确）

4. Agent 存储的核心查询
   - get_conversation_history: 构建 LLM 上下文
   - log_tool_call: 审计和分析
   - get_usage_stats: 用户用量统计

面试速记：
  "Agent 的数据库怎么设计？"
  → SQLite + WAL 模式（开发/小规模）
  → 5 张核心表：users/sessions/messages/tasks/tool_logs
  → tool_calls 用 JSON 字段减少 JOIN
  → FTS5 做对话搜索
"""


def demo_full_storage_workflow():
    """演示完整的 Agent 存储工作流。"""
    print("=" * 60)
    print("  Agent 持久化存储完整演示")
    print("=" * 60)

    init_database()

    storage = AgentStorage()

    # 1. 创建用户
    print("\n  👤 创建用户")
    user = storage.create_user("Alice", "alice@example.com")
    print(f"    用户: {user['name']} ({user['id'][:8]}...)")

    # 2. 创建会话
    print("\n  💬 创建会话")
    session = storage.create_session(user["id"], "学习 AI Agent")
    print(f"    会话: {session['title']} ({session['id'][:8]}...)")

    # 3. 模拟对话（保存消息）
    print("\n  📝 模拟对话")
    messages = [
        ("user", "什么是 AI Agent？", 0, 50),
        ("assistant", "AI Agent 是一种能自主感知、决策、执行的智能系统...", None, 120),
        ("user", "它由哪些组件组成？", 0, 40),
        ("assistant", "主要由 LLM、规划器、记忆系统和工具调用四部分组成。",
         [{"name": "search", "arguments": {"query": "Agent components"}}], 80),
        ("tool", "搜索结果：LLM/规划器/记忆/工具是Agent的核心组件",
         None, 30),
    ]
    for role, content, tool_calls, tokens in messages:
        tc_id = f"call_{hashlib.md5(content.encode()).hexdigest()[:8]}" if role == "tool" else None
        storage.save_message(session["id"], role, content, tool_calls, tc_id, tokens)
        print(f"    [{role:>9s}] {content[:50]}... ({tokens}t)")

    # 4. 记录工具调用日志
    print("\n  🔧 记录工具调用")
    storage.log_tool_call(
        session_id=session["id"],
        tool_name="search",
        input_data={"query": "Agent components"},
        output_data={"results": ["LLM", "规划器", "记忆", "工具"]},
        elapsed_ms=350.5,
        success=True,
    )
    print("    search → 成功 (350ms)")

    # 5. 读取对话历史（Agent 记忆系统用）
    print("\n  📖 读取对话历史（构建 LLM 上下文）")
    history = storage.get_conversation_history(session["id"], max_messages=10)
    print(f"    共 {len(history)} 条消息")
    for msg in history:
        tc_info = ""
        if msg.get("tool_calls_json"):
            tc = json.loads(msg["tool_calls_json"])
            tc_info = f" [tool_call: {tc[0]['name']}]"
        print(f"    [{msg['role']:>9s}] {msg['content'][:60]}...{tc_info}")

    # 6. 会话管理
    print("\n  ⏸️ 暂停会话")
    storage.pause_session(session["id"])
    s = storage.get_session(session["id"])
    print(f"    状态: {s['status']}")

    print("  ▶️ 恢复会话")
    storage.resume_session(session["id"])
    s = storage.get_session(session["id"])
    print(f"    状态: {s['status']}")

    # 7. 任务管理
    print("\n  📋 创建异步任务")
    task = storage.create_task(
        session["id"], "summarize",
        {"max_length": 200}, priority=1,
    )
    print(f"    任务: {task['id'][:8]}... (状态: {task['status']})")

    storage.update_task_status(task["id"], "running")
    storage.update_task_status(
        task["id"], "completed",
        {"summary": "本次对话讨论了 AI Agent 的基本概念和组成..."},
    )
    print(f"    任务完成: 状态 → completed")

    # 8. 查询用户列表
    print("\n  📊 用户会话列表")
    sessions = storage.list_user_sessions(user["id"])
    for s in sessions:
        print(f"    📁 {s['title']} | 消息: {s['message_count']} | Tokens: {s['total_tokens']}")

    # 9. 用量统计
    print("\n  📈 7天用量统计")
    stats = storage.get_usage_stats(user["id"])
    print(f"    会话数: {stats['session_count']}")
    print(f"    总Tokens: {stats['total_tokens']}")
    print(f"    总消息数: {stats['total_messages']}")
    print(f"    工具使用:")
    for tool in stats["tool_usage"]:
        print(f"      {tool['tool_name']}: {tool['call_count']}次, "
              f"平均{tool['avg_latency']:.0f}ms, 失败{tool['failures']}次")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第14章：SQLite + Agent 持久化存储                      ║")
    print("║  Schema设计 · WAL模式 · 会话管理 · 任务状态           ║")
    print("╚══════════════════════════════════════════════════════╝")

    # 清理旧数据库
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("  🧹 已清理旧数据库")

    demo_full_storage_workflow()

    # 展示数据库文件位置
    print(f"\n  💾 数据库文件: {DB_PATH}")
    print(f"  📏 文件大小: {os.path.getsize(DB_PATH):,} bytes")

    print("\n✅ 第14章完成！")
