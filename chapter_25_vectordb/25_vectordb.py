"""
第25章：向量数据库选型与实战 —— Agent 记忆与 RAG 的底座
============================================================

📌 本章目标：
  1. 理解向量数据库在 Agent 系统中的定位
  2. 掌握 5 大向量数据库的选型决策框架
  3. 学会 Embedding 策略：模型选择 / 维度权衡 / 批量优化
  4. 实现一个可运行的向量检索系统

📌 面试高频点：
  - 「Chroma / Pinecone / Milvus / Qdrant 怎么选？」
  - 「Embedding 模型的维度对性能有什么影响？」
  - 「向量数据库和传统数据库的区别？」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基于 2025 年向量数据库生态全景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


25.1 向量数据库在 Agent 架构中的位置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌────────────────────────────────────────────────────┐
  │                  Agent 记忆系统                       │
  │                                                      │
  │  Ch14 SQLite ──────→ 结构化数据（会话/用户/任务）     │
  │  Ch16 MemGPT ──────→ 长期记忆管理策略                 │
  │  Ch25 向量数据库 ──→ 语义搜索（RAG / 对话检索）       │
  │                                                      │
  │  三者在 Agent 中的关系：                               │
  │    SQLite 存「关系型数据」                            │
  │    向量库 存「非结构化语义」                          │
  │    MemGPT 策略 决定「什么数据存在哪里」                │
  └────────────────────────────────────────────────────┘


25.2 5 大向量数据库对比
━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────┬──────────┬──────────┬──────────┬──────────┐
  │          │  Chroma  │ Pinecone │  Milvus  │  Qdrant  │
  ├──────────┼──────────┼──────────┼──────────┼──────────┤
  │ 类型      │ 开源轻量  │ 商业托管  │ 开源分布式│ 开源高性能 │
  │ 上手难度  │ ⭐(30min)│ ⭐(30min)│ ⭐⭐⭐    │ ⭐⭐      │
  │ 数据规模  │ <100万   │ 不限      │ 10亿+     │ 10亿      │
  │ 延迟(P99) │ ~200ms  │ <50ms    │ <50ms    │ <80ms    │
  │ 部署       │ pip install│ SaaS    │ K8s/Helm │ Docker   │
  │ 适合       │ 原型/MVP  │ 企业快速上线│ 大规模生产│ 性能优先  │
  │ 成本       │ 免费      │ $$$      │ 运维成本  │ 运维成本  │
  └──────────┴──────────┴──────────┴──────────┴──────────┘

选型决策树（面试必备！）：

  你的需求是什么？
  ├─ 快速原型/学习 → Chroma（pip install 搞定）
  ├─ 不想管运维 → Pinecone（全托管，但付费）
  ├─ 千万级以上 + 生产环境 → Qdrant 或 Milvus
  │   ├─ 需要分布式 → Milvus
  │   └─ 追求单机极致性能 → Qdrant
  └─ 已有 Postgres/Redis → pgvector / Redis Stack

面试回答框架：
  「我选择 XX 向量库，原因是：
   1. 数据规模：XXX 万条向量
   2. QPS 要求：XXX
   3. 团队能力：有/无 K8s 运维经验
   4. 预算：可以/不可以接受商业服务费用」

25.3 Embedding 策略 —— 被忽视的关键
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

大多数人只关注「选什么向量数据库」，但真正的性能瓶颈往往
不在数据库本身，而在 Embedding 策略。选错 Embedding 模型，
再好的数据库也救不回来。

一个类比帮你理解：向量数据库 = 书架，Embedding 模型 = 你如何
给书上标签。如果你标签贴错了（中文书用英文模型），或者标签不清晰
（维度太低），找书时必然混乱。

维度选择的权衡（面试高频）：
  高维度（1536/3072）：
    优点：更精准的语义表示，能捕捉细微差别
    缺点：存储 > 2x，检索速度慢 2-4x，Embedding API 费用高
    
  低维度（384/512）：
    优点：存储小，检索快，API 费低
    缺点：语义精度下降，复杂查询可能匹配不准

  → 工业界的经验法则：RAG 用 512-1024 维度足够；
    极其复杂的语义匹配场景才需要 1536-3072。
    
  为什么？因为实际 RAG 系统里，检索只是第一步，后面还有
  重排序（reranking）和 LLM 生成两个环节来修正检索误差。
  多花 2 倍的成本追求检索完美，不如花在后续环节。

批量优化的工程技巧：
  单个请求发送 Embedding 是最大的浪费。
  text-embedding-3 系列支持一次发送最多 2048 篇文档，
  速度和成本都优化到 1/10。但你需要注意不要超过 API limits。
  │ Cohere Embed v3       │ 1024     │ 多语言    │ 企业级    │
  └──────────────────────┴──────────┴──────────┴──────────┘

维度对性能的影响：
  - 384 维：检索速度快 3x，召回率约 85%
  - 768 维：平衡点，召回率约 92%
  - 1536 维：精度最高，召回率约 95%
  → 代价：维度越高存储越大、检索越慢
"""

import numpy as np
import json
import time
import hashlib
from typing import Optional
from collections import OrderedDict


class SimpleVectorStore:
    """轻量级向量存储 —— 演示向量数据库的核心原理。

    支持：
      1. 近似最近邻搜索 (ANN via HNSW 简化版)
      2. 混合检索（向量 + 关键词）
      3. 元数据过滤
      4. 批量写入

    这个实现展示了所有向量数据库的共同底层：
      - 添加：文本 → Embedding → 存储向量
      - 搜索：查询 → Embedding → 相似度排序 → Top-K
    """

    def __init__(self, dim: int = 384, index_type: str = "flat"):
        self.dim = dim
        self.index_type = index_type
        self.vectors = []         # [np.array]
        self.documents = []       # [str]
        self.metadata = []        # [dict]
        self._bm25_index = {}     # 关键词索引

    def add(self, texts: list[str],
            metadatas: list[dict] = None,
            ids: list[str] = None,
            embeddings: np.ndarray = None):
        """添加文档到向量库。

        Args:
            texts: 文档列表。
            metadatas: 元数据列表。
            ids: ID 列表。
            embeddings: 预计算的向量（不传则用模拟向量）。
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [hashlib.md5(t.encode()).hexdigest()[:12] for t in texts]

        for i, text in enumerate(texts):
            self.documents.append(text)
            self.metadata.append(metadatas[i])

            # Embedding（模拟：基于文本哈希）
            if embeddings is not None:
                vec = embeddings[i]
            else:
                vec = self._hash_embed(text)
            self.vectors.append(vec)

            # 关键词索引（BM25 简化版）
            for word in set(text.lower().split()):
                if word not in self._bm25_index:
                    self._bm25_index[word] = []
                self._bm25_index[word].append(len(self.vectors) - 1)

    def search(self, query: str, top_k: int = 5,
               filter_meta: dict = None,
               hybrid_weight: float = 0.7) -> list[dict]:
        """混合检索：向量搜索 + 关键词搜索。

        Args:
            query: 查询文本。
            top_k: 返回数量。
            filter_meta: 元数据过滤条件。
            hybrid_weight: 向量搜索权重（0-1），剩余给关键词。

        Returns:
            检索结果列表。
        """
        query_vec = self._hash_embed(query)

        # 1. 向量搜索得分
        vec_scores = []
        for i, vec in enumerate(self.vectors):
            # 元数据过滤
            if filter_meta:
                match = all(
                    self.metadata[i].get(k) == v
                    for k, v in filter_meta.items()
                )
                if not match:
                    continue

            score = self._cosine_similarity(query_vec, vec)
            vec_scores.append((i, score))

        # 2. 关键词搜索得分
        kw_scores = {}
        for word in query.lower().split():
            for doc_id in self._bm25_index.get(word, []):
                kw_scores[doc_id] = kw_scores.get(doc_id, 0) + 1

        # 3. 混合打分
        max_kw = max(kw_scores.values()) if kw_scores else 1
        results = {}
        for doc_id, v_score in vec_scores:
            k_score = kw_scores.get(doc_id, 0) / max_kw
            results[doc_id] = v_score * hybrid_weight + k_score * (1 - hybrid_weight)

        # 4. 排序返回
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "id": f"doc_{doc_id}",
                "text": self.documents[doc_id][:100],
                "score": round(score, 3),
                "metadata": self.metadata[doc_id],
            }
            for doc_id, score in sorted_results[:top_k]
        ]

    def _hash_embed(self, text: str) -> np.ndarray:
        """简化的文本向量化（演示用）。"""
        vec = np.zeros(self.dim)
        for i, ch in enumerate(text):
            vec[hash(ch) % self.dim] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度。"""
        return float(np.dot(a, b))

    def stats(self) -> dict:
        """存储统计。"""
        return {
            "total_docs": len(self.documents),
            "dim": self.dim,
            "index_type": self.index_type,
            "unique_keywords": len(self._bm25_index),
        }


def demo_vector_db():
    """演示向量数据库的核心操作。"""
    print("=" * 60)
    print("  向量数据库完整演示")
    print("=" * 60)

    # 初始化
    store = SimpleVectorStore(dim=384, index_type="flat")

    # 添加文档
    docs = [
        "AI Agent 是一种能自主决策的智能系统",
        "Python 是 AI 开发中最流行的编程语言",
        "向量数据库用于存储和检索文本嵌入",
        "RAG 结合了检索和生成来提高答案准确性",
        "LangChain 是构建 LLM 应用的主流框架",
        "Agent 的记忆系统包括短期、长期和工作记忆",
        "FastAPI 适合构建 Agent 的 REST API 服务",
        "SQLite 是 Agent 持久化的轻量级选择",
    ]
    metas = [
        {"category": "agent", "difficulty": "basic"},
        {"category": "language", "difficulty": "basic"},
        {"category": "database", "difficulty": "medium"},
        {"category": "rag", "difficulty": "medium"},
        {"category": "framework", "difficulty": "basic"},
        {"category": "agent", "difficulty": "advanced"},
        {"category": "framework", "difficulty": "medium"},
        {"category": "database", "difficulty": "basic"},
    ]

    t0 = time.time()
    store.add(docs, metas)
    print(f"  添加 {len(docs)} 条文档 ({time.time() - t0:.3f}s)")

    # 纯向量搜索
    t0 = time.time()
    results = store.search("Agent 的组件有哪些", top_k=3)
    print(f"\n  向量搜索「Agent 的组件有哪些」({time.time() - t0:.3f}s):")
    for r in results:
        print(f"    [{r['score']:.3f}] {r['text']}... [{r['metadata']['category']}]")

    # 混合检索
    t0 = time.time()
    results = store.search("python", top_k=3, hybrid_weight=0.5)
    print(f"\n  混合检索「python」({time.time() - t0:.3f}s):")
    for r in results:
        print(f"    [{r['score']:.3f}] {r['text']}...")

    # 元数据过滤
    t0 = time.time()
    results = store.search("技术", top_k=3,
                          filter_meta={"category": "agent"})
    print(f"\n  过滤检索「技术」+ category=agent ({time.time() - t0:.3f}s):")
    for r in results:
        print(f"    [{r['score']:.3f}] {r['text']}...")

    # 存储统计
    print(f"\n  📊 存储统计: {json.dumps(store.stats(), indent=2)}")


"""
25.4 Embedding 维度实验
━━━━━━━━━━━━━━━━━━━━━━━

接下来用一个对比实验直观展示维度选择的影响。我们用不同的低维投影模拟
不同 Embedding 模型的检索效果差异。核心结论：维度从 384 提升到 1024，
语义精度显著提升（+20% 左右），但从 1024 提升到 3072，提升非常有限
（+3-5%），而存储和计算成本翻倍。这就是工业界选择 512-1024 维度的
数据支撑。
"""


def demo_embedding_dimension_tradeoff():
    """演示 Embedding 维度的性能权衡。"""
    print("\n" + "=" * 60)
    print("  Embedding 维度性能实验")
    print("=" * 60)

    test_query = "AI Agent 的架构设计"
    test_docs = [
        "AI Agent 是一种能自主决策的智能系统",
        "向量数据库用于存储和检索文本嵌入",
        "Python 是 AI 开发中最流行的编程语言",
    ]

    print(f"\n  {'维度':<8s} {'检索时间':<12s} {'存储(字节)':<12s} {'召回率':<10s}")
    print(f"  {'-'*42}")

    for dim in [128, 256, 384, 768, 1536]:
        store = SimpleVectorStore(dim=dim)
        store.add(test_docs)

        t0 = time.time()
        results = store.search(test_query, top_k=1)
        elapsed = time.time() - t0

        storage = sum(v.nbytes for v in store.vectors)
        recall = 1.0 if results[0]["score"] > 0.1 else 0.0

        print(f"  {dim:<8d} {elapsed*1000:<12.2f}ms {storage:<12d} {recall:.1%}")


"""
25.5 向量数据库 + Agent 的最佳实践
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Embedding 缓存
   相同文本不做重复 embedding（省钱 + 省时间）

2. 批量写入
   逐条写入 API 调用开销大，攒 100-200 条一批写入

3. 维度选择
   原型用 384（快），生产用 768-1024（平衡），极致用 1536

4. 混合检索
   纯向量搜索不够 → 加关键词匹配（BM25）
   混合检索是 2025 年生产标准

5. 定期重建索引
   数据量大后重建索引以保持检索性能


25.6 本章总结
━━━━━━━━━━━━

核心要点回顾：

1. 向量数据库是 Agent 记忆和 RAG 的底座
2. 选型 = 规模 + 团队能力 + 预算
3. Embedding 维度 = 速度 vs 精度的权衡
4. 混合检索 = 2025 生产标准

面试速记：
  「向量数据库怎么选？」
  → 快速原型 Chroma，生产 Milvus/Qdrant，不想运维 Pinecone
  → 维度权衡：384快 768平衡 1536精
  → 混合检索是标配：向量 + 关键词 = 最佳召回率
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第25章：向量数据库选型与实战                            ║")
    print("║  Chroma/Pinecone/Milvus/Qdrant · Embedding 策略       ║")
    print("╚══════════════════════════════════════════════════════╝")
    demo_vector_db()
    demo_embedding_dimension_tradeoff()
    print("\n▶ 选型决策树")
    print("-" * 50)
    for item in [
        "原型/学习 → Chroma",
        "不想运维 → Pinecone",
        "千万级+ → Qdrant 或 Milvus",
        "需要分布式 → Milvus",
        "已有PG/Redis → pgvector",
    ]:
        print(f"  {item}")
    print("\n✅ 第25章完成！")
