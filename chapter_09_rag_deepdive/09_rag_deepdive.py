"""
第9章：RAG 原理与实现详解 —— 从 Naive 到 Agentic RAG
=====================================================

📌 本章目标：
  1. 理解 RAG 的完整技术演进路线
  2. 掌握 Naive RAG → Advanced RAG → GraphRAG → Agentic RAG 的原理
  3. 实现一个可运行的 Advanced RAG 系统
  4. 理解 HyDE、Multi-Query、Self-RAG 等高级技术
  5. 了解 RAG 评估体系（RAGAS）

📌 面试高频点：
  - RAG 的核心流程和每步的作用
  - 为什么纯向量搜索不够用？（关键词 vs 语义）
  - GraphRAG 解决了什么问题？
  - Agentic RAG 和传统 RAG 的本质区别
  - 如何评估 RAG 系统的质量？


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9.1 RAG 技术演进全景图
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAG 从 2020 年到 2025 年经历了三个重要阶段：

  ┌──────────────────────────────────────────────────────┐
  │                                                        │
  │  2020-2022: Naive RAG                                 │
  │  「问 → 检索 → 生成」三步走                              │
  │  代表: Facebook RAG 论文、LangChain 基础 QA Chain        │
  │                                                        │
  │  2023-2024: Advanced RAG                              │
  │  加入查询重写、混合检索、重排序、纠错等优化                │
  │  代表: HyDE、Multi-Query、Self-RAG、CRAG               │
  │                                                        │
  │  2024-2025: Next-Gen RAG                              │
  │  知识图谱、Agent 协作、多模态、长文本                     │
  │  代表: GraphRAG (微软)、Agentic RAG、Multimodal RAG     │
  │                                                        │
  └──────────────────────────────────────────────────────┘

  Naive RAG 的问题（面试常问！）:
    1. 检索不准：语义相似 ≠ 问题相关
    2. 上下文丢失：大块文本导致 LLM 忽略关键信息
    3. 无法处理复杂关系：「A影响了B，B又影响了C」需要多跳推理
    4. 无自检能力：检索到无关内容也不知道


9.2 Naive RAG 实现 —— 理解基础再谈优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from typing import Optional


class NaiveVectorStore:
    """最小化的向量存储 —— 帮助理解向量检索的本质。

    面试中常被问「向量检索的原理是什么」，
    这个实现展示了答案：将查询和文档都转为向量，
    通过余弦相似度找到最相似的 Top-K 个。
    """

    def __init__(self):
        self.documents = []         # 原始文档列表
        self.embeddings = []        # 对应的向量列表
        self.chunks = []            # 分块信息

    def add_documents(self, docs: list[str],
                      chunk_size: int = 200,
                      chunk_overlap: int = 50):
        """添加文档：先分块，再「嵌入」（这里用简化模拟表示）。

        Args:
            docs: 文档列表。
            chunk_size: 分块大小（字符数）。
            chunk_overlap: 块间重叠大小。
        """
        for doc in docs:
            for i in range(0, len(doc), chunk_size - chunk_overlap):
                chunk = doc[i:i + chunk_size]
                self.chunks.append(chunk)
                # 用 TF-IDF 启发式模拟向量（实际用 embedding 模型）
                embedding = self._simple_hash_embed(chunk)
                self.embeddings.append(embedding)

        print(f"  已添加 {len(self.chunks)} 个文本块")

    def _simple_hash_embed(self, text: str, dim: int = 64) -> np.ndarray:
        """简化版的文本向量化（仅用于理解原理）。

        实际项目用 sentence-transformers / OpenAI Embeddings API。
        """
        # 基于字符哈希的简单向量表示
        vec = np.zeros(dim)
        for i, ch in enumerate(text):
            vec[hash(ch) % dim] += 1
        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """检索最相似的 Top-K 个文本块。

        Args:
            query: 查询文本。
            top_k: 返回数量。

        Returns:
            (文本块, 相似度分数) 的列表。
        """
        query_vec = self._simple_hash_embed(query)
        scores = []
        for i, emb in enumerate(self.embeddings):
            # 余弦相似度 = 点积（因为已归一化）
            score = float(np.dot(query_vec, emb))
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            results.append((self.chunks[idx], score))
        return results


def demo_naive_rag():
    """演示 Naive RAG 的基本流程。"""
    print("=" * 60)
    print("  Naive RAG 流程演示")
    print("=" * 60)

    store = NaiveVectorStore()
    docs = [
        "Python是一种解释型、面向对象的高级编程语言。"
        "它由 Guido van Rossum 于 1991 年首次发布。"
        "Python 以简洁的语法和强大的标准库而闻名。",

        "AI Agent 是一种能够自主感知环境、做出决策并执行行动的智能系统。"
        "它由 LLM、规划器、记忆系统和工具组成。"
        "LangChain 是构建 Agent 的最流行框架之一。",

        "RAG（检索增强生成）是一种将外部知识检索与 LLM 生成结合的技术。"
        "它通过从知识库中检索相关信息来减少大模型的幻觉问题。"
        "GraphRAG 是 RAG 的最新演进，加入了知识图谱。",

        "PyTorch 是 Facebook 开发的开源深度学习框架。"
        "TensorFlow 是 Google 开发的机器学习平台。"
        "两者都是业界最广泛使用的深度学习工具。",
    ]
    store.add_documents(docs)

    queries = [
        "Python 是谁创建的？",
        "什么是 AI Agent？",
        "RAG 如何减少幻觉？",
    ]
    for q in queries:
        print(f"\n  🔍 查询: {q}")
        results = store.search(q, top_k=2)
        for chunk, score in results:
            print(f"    [{score:.3f}] {chunk[:80]}...")


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9.3 Advanced RAG —— 三板斧优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Advanced RAG 在 Naive RAG 的基础上加入三大优化：
  1. 查询重写 (Query Rewriting)
  2. 混合检索 (Hybrid Search)
  3. 重排序 (Reranking)
"""


class AdvancedRAG:
    """Advanced RAG 系统 —— 包含查询重写、混合检索、重排序。

    架构流程：
      用户查询
         │
         ▼
    ┌──────────────┐
    │ 查询重写       │ ← 用 LLM 扩展/改写查询
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ 混合检索       │ ← BM25 + 向量搜索
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ 重排序         │ ← Cross-Encoder 精细排序
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ LLM 生成       │ ← 注入检索上下文
    └──────────────┘
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: LLM 调用函数（可选）。
        """
        self.llm = llm
        self.store = NaiveVectorStore()
        self.bm25_index = {}  # 简化版 BM25 索引

    def query_rewrite(self, query: str) -> list[str]:
        """查询重写：将一个查询扩展为多个变体。

        策略：
          1. 原始查询
          2. 关键词提取版本
          3. 同义词扩展版本

        在真实项目中，这一步用 LLM 完成。
        比如用 prompt：「将以下查询改写为 3 个不同角度的搜索查询」
        """
        variations = [query]

        # 简单模拟：如果查询中有「怎么」，增加「方法」「教程」
        if "怎么" in query or "如何" in query:
            variations.append(query.replace("怎么", "的方法").replace("如何", "的方法"))

        # 简单模拟：拆分长查询为多个关键词查询
        keywords = [w for w in query.replace("？", "").replace("?", "").split()
                     if len(w) > 1]
        if len(keywords) > 2:
            variations.append(" ".join(keywords[:3]))

        return variations

    def add_documents(self, docs: list[str]):
        """添加文档到混合索引。"""
        self.store.add_documents(docs)
        # 构建简化的 BM25 索引
        for i, chunk in enumerate(self.store.chunks):
            words = chunk.lower().split()
            for word in set(words):
                if word not in self.bm25_index:
                    self.bm25_index[word] = []
                self.bm25_index[word].append(i)

    def hybrid_search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """混合检索：融合向量搜索和关键词搜索的结果。

        Args:
            query: 查询文本。
            top_k: 返回数量。

        Returns:
            融合后的检索结果。
        """
        # 向量搜索结果
        vector_results = self.store.search(query, top_k=top_k)
        vector_scores = {chunk: score for chunk, score in vector_results}

        # 关键词搜索结果（简化版 BM25）
        kw_scores = {}
        for word in query.lower().split():
            if word in self.bm25_index:
                for doc_id in self.bm25_index[word]:
                    chunk = self.store.chunks[doc_id]
                    kw_scores[chunk] = kw_scores.get(chunk, 0) + 1

        # 混合打分：向量分数 * 0.6 + 关键词分数 * 0.4
        all_chunks = set(vector_scores.keys()) | set(kw_scores.keys())
        fused = []
        for chunk in all_chunks:
            v_score = vector_scores.get(chunk, 0)
            k_score = kw_scores.get(chunk, 0)
            # 归一化关键词分数
            max_k = max(kw_scores.values()) if kw_scores else 1
            k_score_norm = k_score / max_k
            final_score = v_score * 0.6 + k_score_norm * 0.4
            fused.append((chunk, final_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    def rerank(self, query: str, candidates: list[tuple[str, float]],
               top_k: int = 3) -> list[tuple[str, float]]:
        """重排序：对候选结果进行精细排序。

        在真实项目中，这一步用 Cross-Encoder 模型（如 BGE-reranker）：
          - 对每个 (query, chunk) 对打分
          - 与向量搜索的「分别打分」不同，Cross-Encoder 同时看两边

        这里用简化的规则模拟：
          - 查询关键词在块中出现得越早，分数越高
          - 块越短（越聚焦），分数越高
        """
        query_lower = query.lower()
        reranked = []
        for chunk, orig_score in candidates:
            chunk_lower = chunk.lower()

            # 规则1: 关键词位置分（越靠前越好）
            position_score = 0.0
            for word in query_lower.split():
                pos = chunk_lower.find(word)
                if pos >= 0:
                    position_score += max(0, 1.0 - pos / len(chunk_lower))

            # 规则2: 长度分（越短越聚焦）
            length_score = max(0, 1.0 - len(chunk) / 1000)

            # 综合分（在真实项目中由 Cross-Encoder 输出）
            final = orig_score * 0.5 + position_score * 0.3 + length_score * 0.2
            reranked.append((chunk, final))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9.4 GraphRAG —— 知识图谱 + RAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

为什么需要 GraphRAG？

  Naive RAG 的场景：「Python 是什么？」
    → 检索到「Python是一种编程语言」的文本块
    → 回答正确 ✓

  Naive RAG 失败的场景：「Python 的设计者还参与了哪些项目？」
    → 检索到「Python是由Guido创建的」和「Dropbox使用Python」
    → 但这些块之间没有关联，无法回答「Guido还做过什么」
    → 回答失败 ✗

  GraphRAG 的优势：
    → 知识图谱中存储了「Guido → 创建 → Python」
    → 以及「Guido → 工作于 → Google → 参与 → Dropbox」
    → 可以沿着图谱的边（关系）进行多跳推理
    → 回答正确 ✓

GraphRAG 的核心原理：

  1. 图构建 (Graph Construction)
     从文档中抽取实体和关系，构建知识图谱

  2. 混合检索 (Hybrid Retrieval)
     同时使用向量搜索和图遍历

  3. 多跳推理 (Multi-hop Reasoning)
     沿着图谱的边寻找推理路径
"""


class SimpleKnowledgeGraph:
    """简化的知识图谱 —— 演示 GraphRAG 的核心概念。

    节点: 实体（人、组织、概念）
    边: 关系（创建、工作于、属于）
    """

    def __init__(self):
        self.graph = {}  # {entity: {relation: [target_entities]}}

    def add_relation(self, subject: str, relation: str, obj: str):
        """添加三元组：主语-关系-宾语。

        Args:
            subject: 主语实体。
            relation: 关系类型。
            obj: 宾语实体。
        """
        if subject not in self.graph:
            self.graph[subject] = {}
        if relation not in self.graph[subject]:
            self.graph[subject][relation] = []
        self.graph[subject][relation].append(obj)

    def traverse(self, start: str, max_depth: int = 2) -> list[str]:
        """从起始节点出发进行图遍历。

        Args:
            start: 起始实体。
            max_depth: 最大遍历深度。

        Returns:
            到达的所有实体列表。
        """
        visited = set()
        to_visit = [(start, 0)]
        while to_visit:
            node, depth = to_visit.pop(0)
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            if node in self.graph:
                for relation, targets in self.graph[node].items():
                    for target in targets:
                        to_visit.append((target, depth + 1))
        return list(visited)

    def find_path(self, start: str, end: str, max_depth: int = 3) -> Optional[list]:
        """寻找两个实体之间的路径（简化版 BFS）。

        Args:
            start: 起始实体。
            end: 目标实体。
            max_depth: 最大搜索深度。

        Returns:
            路径列表，未找到返回 None。
        """
        queue = [(start, [start])]
        visited = {start}
        while queue:
            node, path = queue.pop(0)
            if len(path) > max_depth + 1:
                continue
            if node == end and len(path) > 1:
                return path
            if node in self.graph:
                for relation, targets in self.graph[node].items():
                    for target in targets:
                        if target not in visited:
                            visited.add(target)
                            queue.append((target, path + [target]))
        return None


def demo_graphrag():
    """演示 GraphRAG 的多跳推理能力。"""
    print("=" * 60)
    print("  GraphRAG 多跳推理演示")
    print("=" * 60)

    kg = SimpleKnowledgeGraph()

    # 构建知识图谱
    kg.add_relation("Guido van Rossum", "创建", "Python")
    kg.add_relation("Guido van Rossum", "工作于", "Google")
    kg.add_relation("Guido van Rossum", "工作于", "Dropbox")
    kg.add_relation("Guido van Rossum", "工作于", "Microsoft")
    kg.add_relation("Python", "用于", "AI Agent")
    kg.add_relation("Python", "用于", "数据分析")
    kg.add_relation("AI Agent", "依赖", "LLM")
    kg.add_relation("LLM", "包括", "GPT-4")
    kg.add_relation("LLM", "包括", "Claude")
    kg.add_relation("Google", "开发", "TensorFlow")
    kg.add_relation("Microsoft", "投资", "OpenAI")
    kg.add_relation("OpenAI", "开发", "GPT-4")

    print("\n  知识图谱已构建:")
    for subj, relations in kg.graph.items():
        for rel, objs in relations.items():
            for obj in objs:
                print(f"    {subj} --[{rel}]--> {obj}")

    # 多跳推理示例
    queries = [
        ("Guido van Rossum", "TensorFlow"),  # Guido → Google → TensorFlow
        ("Python", "GPT-4"),                 # Python → AI Agent → LLM → GPT-4
        ("Guido van Rossum", "OpenAI"),      # Guido → Microsoft → OpenAI
    ]

    print("\n  多跳推理测试:")
    for start, end in queries:
        path = kg.find_path(start, end)
        if path:
            print(f"    {start} → {end}: {' → '.join(path)}")
        else:
            print(f"    {start} → {end}: 未找到路径")


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9.5 Agentic RAG —— 让 RAG 自己思考
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统 RAG 是固定流水线：
  「不管什么问题，都走 检索→生成 两步」

Agentic RAG 是有判断力的系统：
  「先想想这个问题需要什么，再决定怎么做」

核心能力升级：

  ┌──────────────┬──────────────────────┬──────────────────────┐
  │     能力      │      传统 RAG         │     Agentic RAG      │
  ├──────────────┼──────────────────────┼──────────────────────┤
  │ 查询处理      │ 单次检索              │ 多轮自适应检索         │
  │ 检索策略      │ 固定（总是向量搜索）   │ 动态选择（向量/图谱/API）│
  │ 结果判断      │ 无（直接用检索结果）   │ 自检 + 补充检索         │
  │ 工具使用      │ 只有检索              │ 检索 + 计算 + API     │
  │ 错误处理      │ 无                    │ 多轮重试 + 纠错        │
  └──────────────┴──────────────────────┴──────────────────────┘

Agentic RAG 的工作流程:

  用户: "2024年OpenAI的营收比2023年增长了百分之多少？"

  传统 RAG:
    → 检索「OpenAI 营收」→ 可能找到 2023 和 2024 的数据
    → 但 RAG 不会算百分比，直接给了用户两段数字
    → 用户还需要自己算

  Agentic RAG:
    → Step 1: 分析问题 → 需要「2023营收」「2024营收」「百分比计算」
    → Step 2: 搜索「OpenAI 2023年营收」 → 得到 $1.6B
    → Step 3: 搜索「OpenAI 2024年营收」 → 得到 $3.7B
    → Step 4: 计算 (3.7-1.6)/1.6 = 131.25%
    → Step 5: 回答「增长约 131%」

  Agentic RAG 包含了一个 Agent 循环：
    思考需要什么信息 → 检索/计算 → 不满足就继续 → 满足则输出

业内数据（2025年）：
  - Agentic RAG 在复杂查询上的准确率比传统 RAG 高 30-50%
  - 但 90% 的 Agentic RAG 项目在生产中失败！
  - 失败原因：过度设计、缺少评估、成本失控
"""


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9.6 RAG 评估体系 —— 如何衡量 RAG 质量？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAGAS (RAG Assessment) 是最流行的 RAG 评估框架。

核心指标：
  ┌──────────────────┬────────────────────────────────────┐
  │      指标         │              含义                   │
  ├──────────────────┼────────────────────────────────────┤
  │ Faithfulness     │ 回答是否完全基于检索到的上下文？       │
  │ Answer Relevance │ 回答是否直接回应了问题？              │
  │ Context Recall   │ 检索到的内容覆盖了回答所需的全部信息？ │
  │ Context Precision│ 检索到的内容有多少是真正相关的？       │
  │ Answer Correctness│ 答案的事实准确性                     │
  └──────────────────┴────────────────────────────────────┘

2025年评估趋势：
  - LLM-as-Judge 成为主流（用 GPT-4 评估 RAG 输出）
  - 人工标注仍然重要（作为 Ground Truth）
  - 多维度评估 > 单一指标
"""


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9.7 本章总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心要点回顾：

1. Naive RAG = 问 → 检索 → 生成
   问题：检索不准、无自检、无法多跳推理

2. Advanced RAG = Naive + 查询重写 + 混合检索 + 重排序
   面试重点：为什么混合检索比纯向量好？
   → 向量擅长语义，但不懂精确关键词匹配
   → 法律/金融/监管领域，精确匹配和语义理解同等重要

3. GraphRAG = RAG + 知识图谱
   核心价值：多跳推理能力
   适用场景：关系密集型问题（「A影响了B，B又影响了C」）

4. Agentic RAG = RAG + Agent 循环
   核心价值：动态决策 → 自主判断需要什么信息
   注意：90% 的生产失败率 — 从简单开始，逐步迭代

面试速记：
  "RAG 有什么高级技术？"
  → 查询重写(HyDE) → 混合检索(Hybrid) → 重排序(Reranker)
  → GraphRAG(知识图谱) → Agentic RAG(Agent 循环)
  → 每个技术解决一个具体痛点，不要盲目叠加
"""


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  第9章：RAG 原理与实现详解                             ║")
    print("║  Naive RAG → Advanced → GraphRAG → Agentic RAG      ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n▶ 9.2 Naive RAG 演示")
    demo_naive_rag()

    print("\n▶ 9.3 Advanced RAG 演示")
    rag = AdvancedRAG()
    docs = [
        "Python是一种由Guido van Rossum创建的编程语言，"
        "首次发布于1991年。它以简洁的语法风格著称。",

        "AI Agent能自主感知环境、做出决策并执行行动。"
        "它集成了LLM、规划器、记忆系统和工具调用。",

        "RAG通过从知识库检索相关信息来增强LLM的生成能力，"
        "有效减少模型的幻觉问题。GraphRAG加入了知识图谱。",
    ]
    rag.add_documents(docs)

    query = "如何用Python构建AI Agent？"
    print(f"\n  🔍 原始查询: {query}")

    variations = rag.query_rewrite(query)
    print(f"  ✏️ 查询变体: {variations}")

    for v in variations:
        results = rag.hybrid_search(v, top_k=2)
        print(f"\n  查询变体「{v}」的混合检索结果:")
        for chunk, score in results:
            print(f"    [{score:.3f}] {chunk[:100]}...")

        reranked = rag.rerank(v, results, top_k=2)
        print(f"  重排序后:")
        for chunk, score in reranked:
            print(f"    [{score:.3f}] {chunk[:100]}...")

    print("\n▶ 9.4 GraphRAG 演示")
    demo_graphrag()

    print("\n▶ 9.5 Agentic RAG 概念")
    print("-" * 50)
    print("核心区别：")
    print("  传统RAG: 固定流水线「检索→生成」")
    print("  Agentic RAG: 动态决策 「思考→检索→判断→再检索→...」")
    print("")
    print("2025年行业数据：")
    print("  准确率提升 30-50%，但 90% 生产失败率")
    print("  建议：从简单开始，逐步迭代，持续评估")

    print("\n▶ 9.6 RAGAS 评估体系")
    print("-" * 50)
    metrics = [
        ("Faithfulness", "回答是否基于检索到的内容？"),
        ("Answer Relevance", "回答是否直接回应了问题？"),
        ("Context Recall", "检索是否覆盖了所有必要信息？"),
        ("Context Precision", "检索到的内容有多少是相关的？"),
    ]
    for name, desc in metrics:
        print(f"  {name}: {desc}")

    print("\n✅ 第9章完成！")
