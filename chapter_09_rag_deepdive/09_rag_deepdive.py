"""
第9章：RAG 深度解析 —— 从 Naive 到生产级，面试到工程全覆盖
===========================================================

内容核对：2026-08-01
说明：标注为模拟的实现与数值用于讲解概念，不代表真实 SDK、协议或基准结果。

📌 本章目标：
  1. 理解 RAG 的完整技术演进路线（Naive → Advanced → GraphRAG → Agentic）
  2. 掌握 Chunk 切分策略的工程选型（固定/语义/递归 + 评测驱动调参）
  3. 掌握 Embedding 模型与维度的评测方法
  4. 深入理解混合检索 RRF 融合算法和 Cross-Encoder 重排序原理
  5. 实现可运行的 Advanced RAG 和 GraphRAG 系统
  6. 掌握 RAGAS 评估体系和 4 种生产失败模式的应对策略

📌 面试高频点：
  - Chunk 怎么切分？为什么这个大小？（本章核心深度！）
  - Embedding 模型怎么选？维度怎么定？
  - 为什么混合检索比纯向量好？RRF 怎么做？
  - Bi-Encoder 和 Cross-Encoder 的本质区别？
  - GraphRAG 解决了什么纯向量 RAG 解决不了的问题？
  - Agentic RAG 和传统 RAG 的本质区别
  - RAG 上线后最常见的 4 种失败模式及对策


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


9.1.1 Chunk 切分策略深度 —— RAG 最被低估的环节
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

面试官问「chunk 怎么切分？为什么这样切？」——这是在考察你是否真正
做过生产级 RAG，而不仅仅是跑过 LangChain 的 demo。

Chunk 切分是 RAG Pipeline 的**第一步**，也是**最致命的瓶颈**。
切错了 chunk，后面的检索、重排序、LLM 生成全都会受影响——就像
地基歪了大楼必然歪。

▍ 三种主流切分策略

┌─────────────────┬────────────────────────┬──────────────────────┐
│ 策略              │        原理             │       适用场景        │
├─────────────────┼────────────────────────┼──────────────────────┤
│ 固定大小切分      │ 按字符/token 等长切     │ 通用文档、入门级       │
│ 语义切分          │ 按段落/句子边界 + 相似度  │ 技术文档、论文         │
│ 递归字符切分      │ 先按分隔符层级切          │ LangChain 默认方案     │
└─────────────────┴────────────────────────┴──────────────────────┘

▍ chunk_size 选择的工程经验（面试高分点）

  太小（100-200 字符）:
    优点 → 检索精度高，向量信号集中
    致命问题 → 丢失上下文！「Python 在...之后发布了...」只剩半句话
    
  中等（500-1000 字符）:
    RAG 的「甜点区」—— 保持语义完整性的同时控制噪声
    一个 chunk ≈ 一段话，正好是一个完整语义单元
    
  太大（2000+ 字符）:
    优点 → 上下文丰富，LLM 生成质量高
    致命问题 → 检索信噪比低，Embedding 向量被稀释
    「大海捞针」现象：关键词匹配到了，但淹没在 2000 字中

  可作为首轮实验的候选值（不是跨项目结论）:
    技术文档 → 先按标题、段落和完整代码块切分
    客服知识库 → 优先保持一组问答完整
    法律合同 → 优先保持条款及其引用关系完整
    通用语料 → 同时测试多个 chunk_size，再按端到端指标选择

▍ chunk_overlap 不是「加越多越好」

  overlap = chunk 之间的重叠字符/Token 数。
  
  它的真正作用：防止关键信息恰好落在两个 chunk 的边界处。
  比如「Python 是 Guido van Rossum 在 CWI 工作时创建的」被切成：
    Chunk A: 「Python 是 Guido van Rossum 在」 
    Chunk B: 「CWI 工作时创建的」
    → 两个 chunk 都丢失了完整语义！
  
  正确做法：把 overlap 当作待评测参数，而不是固定百分比。
  过大 → 存储膨胀、检索重复；过小 → 边界信息丢失。
  记录切分器版本，并同时比较检索召回、重复率、延迟和最终答案质量。

▍ 语义切分（Semantic Chunking）—— 面试中的进阶话题

  固定大小切分的根本缺陷：完全无视文档的**自然语义边界**。
  
  语义切分的工作原理：
    1. 把文档按句子拆分
    2. 计算相邻句子的 Embedding 相似度
    3. 在相似度「骤降」处切分 —— 这通常是段落/话题的自然边界
    
  为什么更好？因为 Embedding 模型在「同一话题的句子」上相似度高，
  在「话题切换」处相似度骤降。利用这个特性，切出来的每个 chunk
  天然是语义自洽的，不需要人工调 chunk_size。

  语义切分的代价：
    - 需要对每个句子做 Embedding（增加预处理成本）
    - Chunk 大小不均（有的 150 字，有的 800 字）
    - 不符合 LLM 的「固定 Token 预算」预期

  工程上的折中：先语义切分确定段落边界 + 再在边界内做固定大小切分。
  这也是 LangChain SemanticChunker 的实现方式。

▍ 特殊场景的切分策略

  代码文档 → 按函数/类切分（AST 解析），不是按行
  表格数据 → 保留完整表格在一个 chunk 中，不要横向切
  多语言混合 → 按语言检测 + 独立切分，避免中英文混在一个 chunk


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
9.2.1 Embedding 模型选型 —— 选错模型比选错数据库更致命
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

很多工程师花大量时间纠结「用 Chroma 还是 Pinecone」，但真正的精度瓶颈
在 Embedding 模型上。数据库只是存储，Embedding 模型决定了检索的「视力」。

▍ OpenAI vs 开源 Embedding：到底怎么选？

  text-embedding-3-small（OpenAI）:
    维度: 1536（默认），可通过 dimensions 参数缩短
    费用: 按输入 token 计费，运行前查看官方价格页
    优势: 调试方便、无需部署、中文效果好
    权衡: 维度主要影响向量存储、索引内存和检索质量，不直接改变输入 token 数
    
  bge-large-zh-v1.5（智源，开源）:
    维度: 1024
    费用: 无 API 单价，但需要计算、部署和运维资源
    优势: 中文专用、可微调、无网络依赖
    陷阱: 需要 GPU（4GB VRAM）、部署运维成本
    
  multilingual-e5-large（微软）:
    维度: 1024
    特点: 多语言检索基线之一；是否适用需在本领域语料上评测
    特殊要求: 查询前必须加 "query: " 前缀，文档前加 "passage: "

▍ 维度选择的工程方法

  在同一模型、同一语料和同一索引参数下，对候选维度测量：
    - Recall@k / nDCG@k：是否找到正确文档
    - p50 / p95 延迟：索引与网络开销
    - 索引体积和常驻内存
    - 生成答案的端到端正确率

  不存在跨领域通用的「最佳维度」。先确定可接受的质量下限，再选择满足
  下限的最小维度；任何固定 Recall 数字都必须附数据集、切分和随机种子。

▍ 什么时候需要微调 Embedding 模型？

  如果你做的是垂直领域 RAG（医疗、法律、金融），先建立领域评测集，
  只有通用模型未达标时再比较领域模型或微调方案。
  
  原因：通用 Embedding 模型在垂直术语上的表现很差。
  比如 text-embedding-3 对「心梗」和「心肌梗死」算出的相似度可能
  只有 0.6，但对人类来说这是同一个意思。

  微调数据准备：
    - 正例对：(query, 正确文档 chunk) × 500-2000 条
    - 负例对：(query, 随机文档 chunk) × 同数量
    - 训练方式：对比学习（Contrastive Learning），不是普通微调
    
  效果必须在留出的领域测试集上报告，并与更换模型、混合检索和重排基线比较。


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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9.3.1 混合检索的数学 —— RRF 为什么比加权求和好
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

上面 AdvancedRAG.hybrid_search 用的是简单的加权求和（向量分×0.6 +
关键词分×0.4）。这在 demo 中能跑，但在生产中有严重问题：

  问题 1：两个打分器的分数范围不同
    向量相似度范围 [0, 1]，BM25 分数范围 [0, +∞)（可以到 20、50）
    直接加权 → BM25 分数会「淹没」向量分数

  问题 2：无法确定最佳权重
    0.6/0.4 是拍脑袋的权重。换一个数据集/Embedding 模型，最优权重
    可能变成 0.3/0.7。每次都要重新调参 → 不可维护

▍ RRF (Reciprocal Rank Fusion) —— 工业界标准方案

  RRF 不融合「分数」，而是融合「排名」。公式：

    RRF_score(d) = Σ 1/(k + rank_i(d))

    其中 rank_i(d) 是文档 d 在第 i 个检索器中的排名，
    k 是平滑常数；60 是常见起始值，但仍应在自己的验证集上确认。

  为什么比加权求和好？
    1. 与分数量纲无关 —— 不管 BM25 返回 0.3 还是 50，只看排名
    2. 自动均衡 —— 排名靠前的文档贡献大（1/61 ≈ 0.016 vs 1/90 ≈ 0.011）
    3. 无需调权重 —— 唯一的超参数 k 对结果不敏感（30-100 都可以）
    4. 可扩展 —— 3 个、5 个检索器随便加，不需要重新配权重

  RRF 的一个关键细节：k 值越大，排名的「边际差异」越小。
  k=0 时，排名 1 贡献 1.0，排名 10 贡献 0.1（差异 10 倍）
  k=60 时，排名 1 贡献 0.016，排名 10 贡献 0.014（差异只有 15%）
  → 较大的 k 会让融合更「民主」，不只关注第 1 名

▍ 什么时候混合检索最有价值？

  混合检索不是银弹。它的最大价值在于处理「语义相近但关键词不同」和
  「关键词相同但语义不同」两种矛盾场景。

  必用混合检索的场景：
    法律合同 → 「不可抗力」和「天灾」语义相同但词不同 → 向量检索救命
    技术文档 → 「API」在不同上下文含义完全不同 → 关键词检索救命
    
  可以只用向量的场景：
    科普文章 → 语义检索已经足够
    纯 FAQ → 用户问法高度集中在 200 个模板 → 关键词就够了


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9.3.2 重排序 (Reranking) 深度 —— Cross-Encoder 到底做了什么
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

重排序经常是值得优先验证的一步，但收益与成本取决于候选集、reranker、
硬件和数据集。不要引用跨项目固定百分比；用消融实验比较是否启用重排序。

▍ Bi-Encoder vs Cross-Encoder —— 面试必问

  Bi-Encoder（向量检索用的就是这种）:
    原理: 分别对 query 和 document 编码 → 然后算相似度
    速度: 快（document 向量可以预计算、缓存）
    精度: 中等（query 和 document 在编码阶段「没见过」对方）
    用途: 初筛（从 100 万文档中找前 100 个候选）

  Cross-Encoder:
    原理: 把 query 和 document 拼接在一起 → 联合编码 → 输出一个分数
    速度: 慢（每个 (query, doc) 对都要完整推理一次）
    精度: 高（query 和 document 在编码阶段「互相感知」）
    用途: 精排（从 100 个候选中找前 5 个）

  类比：
    Bi-Encoder = 分别看两个人的简历 → 猜他们合不合适
    Cross-Encoder = 让两个人坐下来聊 → 真实判断合不合适
    
  Cross-Encoder 能捕捉到的 Bi-Encoder 捕捉不到的东西：
    « 精确匹配 » "Python" 和 "Python语言" → Bi-Encoder 相似度可能只有 0.5
    → Cross-Encoder 看到完整上下文后直接打 0.95
    « 否定/反转 » "不是Python" → Bi-Encoder 看不出否定语义
    → Cross-Encoder 能识别否定词降低了相关性

▍ 重排序的成本收益计算

  假设 RAG pipeline 中有大量文档：
    Stage 1 (Bi-Encoder): 全库 → 一批候选（可预计算向量）
    Stage 2 (Cross-Encoder): 候选 → 少量最终结果（逐对联合编码）
    Stage 3 (LLM 生成): 将最终文档作为上下文 → 回答

  如果不加重排序，直接把 Stage 1 的 top-5 喂给 LLM：
    会省掉重排序计算，但可能把排序靠前却不真正相关的文档交给 LLM。
    是否影响最终答案、影响多大，只能通过本领域评测集确认。

  结论：同时报告无重排序/有重排序的 Recall@k、答案正确率、p95 延迟与成本，
  达到预先定义的质量门槛后再决定是否上线。

▍ 主流 Reranker 模型选择

  BGE-Reranker-v2-m3（智源）:
    特点: 多语言，中文效果好，免费
    适合: 通用中文 RAG
    
  托管 Rerank API:
    特点: 无需自建推理服务，模型和价格会随版本变化
    适合: 希望快速验证重排序价值的团队
    选型: 查看当前官方模型卡和价格，并在私有验证集上比较
    
  cross-encoder/ms-marco-MiniLM-L-6-v2:
    特点: 可本地部署的经典轻量基线
    适合: 需要可控部署或建立延迟基线的场景


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

  用户: "某公司2024年营收比2023年增长了百分之多少？"

  传统 RAG:
    → 检索「公司营收」→ 可能找到 2023 和 2024 的数据
    → 但 RAG 不会算百分比，直接给了用户两段数字
    → 用户还需要自己算

  Agentic RAG:
    → Step 1: 分析问题 → 需要「2023营收」「2024营收」「百分比计算」
    → Step 2: 搜索「公司 2023年营收」 → 假设得到 1.6B
    → Step 3: 搜索「公司 2024年营收」 → 假设得到 3.7B
    → Step 4: 计算 (3.7-1.6)/1.6 = 131.25%
    → Step 5: 回答「增长约 131%」

  Agentic RAG 包含了一个 Agent 循环：
    思考需要什么信息 → 检索/计算 → 不满足就继续 → 满足则输出

工程结论：
  - Agentic RAG 可能改善需要多步检索、验证或计算的任务
  - 它也会增加调用次数、延迟和错误路径，不应默认替代固定检索流水线
  - 用同一测试集比较准确率、总 token、p95 延迟和失败率后再决定是否采用
"""


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9.6 RAG 评估体系 —— 如何衡量 RAG 质量？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAGAS (RAG Assessment) 是最流行的 RAG 评估框架，但你不需要把所有
指标都用上。面试时关键是说出「什么场景重点关注哪个指标」。

核心指标及使用优先级：
  ┌──────────────────┬────────────────────────────────────┬──────────┐
  │      指标         │              含义                   │  使用频率 │
  ├──────────────────┼────────────────────────────────────┼──────────┤
  │ Faithfulness     │ 回答是否完全基于检索到的上下文？       │ ⭐⭐⭐⭐⭐ │
  │ Answer Relevance │ 回答是否直接回应了问题？              │ ⭐⭐⭐⭐   │
  │ Context Recall   │ 检索到的内容覆盖了回答所需的全部信息？ │ ⭐⭐⭐⭐   │
  │ Context Precision│ 检索到的内容有多少是真正相关的？       │ ⭐⭐⭐     │
  │ Answer Correctness│ 答案的事实准确性                     │ ⭐⭐⭐     │
  └──────────────────┴────────────────────────────────────┴──────────┘

实战中的优先级逻辑：
  如果你的 RAG 还处于「经常胡说」阶段 → 先优化 Faithfulness
  如果你发现「回答偏题了」→ 重点看 Answer Relevance
  如果「检索结果质量不稳定」→ 查 Context Recall 和 Precision

▍ RAGAS 的实际工程经验

  1. LLM-as-Judge 会产生调用费用与评审偏差
     在 CI 中运行小型哨兵集，完整评测按计划或模型/提示词变更触发

  2. 人工标注 Ground Truth 是必须的 —— LLM 自己评自己会过拟合
     对覆盖关键风险和典型流量的样本做人工标注，作为「锚点」校准 LLM 评分；
     样本量由误差预算和分层覆盖决定，不使用固定百分比

  3. 单次高分不代表上线稳 —— 按文档、模型和提示词的变更风险触发回溯评测，
     因为你的文档在变、Embedding 模型在更新、用户问法在迁移

截至 2026-08-01 的评估重点：
  - LLM-as-Judge 需要用人工锚点校准，并报告模型与 rubric 版本
  - RAGAS 支持在线评估（用户反馈 → 实时指标更新）
  - 多轮对话评估成为新方向（不只是单轮 QA）


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9.6.1 RAG 在生产中的常见失败模式与对策
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

面试官问你「RAG 上线后遇到过什么问题？」——这是在考察你是否真的
把 RAG 部署到了生产环境。

▍ 失败模式 1：检索到无关内容，LLM 照单全收

  现象：用户问「Python 的创始人是谁」，检索到了「Python 是一种编程语言...
  Python 的语法简洁...」，LLM 回答「Python 的创始人是语法简洁的...」← 胡编

  根因：缺乏「拒答机制」——LLM 默认相信检索结果

  对策：
    a) 添加 Faithfulness 检查，阈值由人工标注集校准
    b) 在 System Prompt 中明确指示「如果检索到的信息不包含答案，请直接说不知道」
    c) 设置检索分数阈值，低于阈值的 chunk 不传给 LLM

▍ 失败模式 2：关键信息分散在多个 chunk 中

  现象：一个完整答案被切成 3 个 chunk：「Python 于 1991 年发布」「由 Guido
  van Rossum 创建」「受 ABC 语言影响」。每个 chunk 单独看都不完整。

  根因：chunk 太小或 overlap 不足

  对策：
    a) 在验证集上比较更大的 chunk 或结构化切分
    b) 使用父文档检索策略（检索小块 → 返回大块邻居）
    c) 调整 overlap，并监控边界遗漏与重复检索

▍ 失败模式 3：RAG 成本失控

  现象：一段时间后发现 Embedding 与索引成本高于生成调用

  根因：预处理时对全量文档做了 Embedding + 线上每次查询都从头调 Embedding API

  对策：
    a) 按 provider 当前批处理限制批量化，不逐条调用 API
    b) Embedding 结果缓存到本地（S3 / 本地文件 / 向量库自带的持久化）
    c) 热查询缓存（记录命中率、陈旧率和失效策略）
    d) 比较托管与自部署 Embedding 的总拥有成本，而不只看 API 单价

▍ 失败模式 4：向量库索引腐烂

  现象：上线一段时间后质量持续下降

  根因：知识库更新了，但向量索引没有重建 → 部分 chunk 指向已过时的内容

  对策：
    a) 建立文档变更 Hook → 自动重建受影响 chunk 的 Embedding
    b) 给每个 chunk 加版本号和过期时间戳
    c) 根据变更量、索引一致性和恢复目标安排全量校验或重建
"""


"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9.7 本章总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心要点回顾：

1. Naive RAG = 问 → 检索 → 生成
   问题：检索不准、无自检、无法多跳推理

2. Chunk 切分（面试高频！）
   核心认知：没有「万能 chunk_size」，根据文档类型选择
   面试标准回答：「按文档结构保留语义单元，用验证集比较多个候选大小」
   进阶：语义切分 + 固定大小切分的混合策略
   overlap 与 chunk_size 都是需要通过检索和端到端评测确定的参数

3. Embedding 模型选型
   托管基线 → 当前可用的 Embedding API（记录模型快照和 dimensions）
   中文/本地基线 → 在目标语料上评测合适的开源多语言或中文模型
   垂直领域 → 只有基线达不到质量门槛时再考虑微调
   维度选择：在质量下限内选择存储、延迟和成本合适的最小维度

4. Advanced RAG = Naive + 查询重写 + 混合检索(RRF) + 重排序(Cross-Encoder)
   面试重点：为什么混合检索比纯向量好？
   → 向量擅长语义但不擅长关键词精确匹配
   → 法律/金融领域两者的互补性尤其明显
   → RRF 避免直接混合异构分数，k=60 可作起点但仍要验证

5. 重排序是应优先做消融实验的优化
   Cross-Encoder 的质量收益和额外延迟必须在目标数据与硬件上测量
   面试答法：「我使用 Bi-Encoder 做初筛 + Cross-Encoder 做精排的两阶段检索」

6. GraphRAG = RAG + 知识图谱
   核心价值：多跳推理能力
   适用场景：关系密集型问题（「A影响了B，B又影响了C」）

7. Agentic RAG = RAG + Agent 循环
   核心价值：动态决策 → 自主判断需要什么信息
   注意：增加调用、延迟和错误路径 — 从简单基线开始，逐步迭代

8. RAG 生产 Checklist
   ✅ 检索分数阈值 → 拒绝低质量 chunk
   ✅ 用人工锚点校准 Faithfulness 阈值
   ✅ Embedding 批量预处理 + 持久化缓存
   ✅ 文档变更 Hook → 自动增量更新索引
   ✅ 热查询缓存 → 记录命中率、陈旧率和真实成本变化
   ✅ 按知识库/模型/提示词变更风险触发回溯评测

面试速记（完整版）:
  "RAG 核心技术栈？"
  → 切分策略(固定/语义) → Embedding 模型(text-embedding/bge/e5)
  → 混合检索(RRF融合BM25+向量) → 重排序(Cross-Encoder精排)
  → 查询重写(HyDE/Multi-Query) → GraphRAG(知识图谱)
  → Agentic RAG(Agent循环) → 评估(RAGAS) → 生产监控
  "不要盲目叠加技术，每个技术解决一个具体痛点"
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
    print("工程建议：")
    print("  用同一评测集比较固定 RAG 与 Agentic RAG 的质量、成本和延迟")
    print("  从简单流水线开始，只有证据支持时才增加 Agent 循环")

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
