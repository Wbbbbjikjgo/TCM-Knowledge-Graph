# 中医药知识图谱智能问答与内容生成系统

> 基于知识图谱 + RAG 的中医药垂直领域智能问答与科普内容自动生成系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]() [![Neo4j](https://img.shields.io/badge/Neo4j-4.4%2B-green)]() [![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)]() [![LLM](https://img.shields.io/badge/LLM-DeepSeek-orange)]()

---

## 📖 目录

- [1. 项目概述](#1-项目概述)
- [2. 技术架构](#2-技术架构)
- [3. 技术栈详解](#3-技术栈详解)
- [4. 模型国产化与易下载说明](#4-模型国产化与易下载说明)
- [5. 核心功能与实现方案](#5-核心功能与实现方案)
- [6. 数据库设计](#6-数据库设计)
- [7. 项目结构](#7-项目结构)
- [8. 部署指南](#8-部署指南)
- [9. 性能指标](#9-性能指标)
- [10. 项目路线图](#10-项目路线图)

---

## 1. 项目概述

### 1.1 项目背景

中医药领域知识体系庞大复杂，包含**药材、方剂、症状、证型、功效**等多维度实体及其间错综复杂的语义关系。传统基于关键词检索的方式难以捕捉“辨证论治”的深层逻辑关联，而通用大语言模型在中医药垂直领域容易出现“幻觉”，生成缺乏事实依据的回答。

本项目构建一个**融合知识图谱与检索增强生成（RAG）** 的中医药智能问答与内容生成系统，通过将结构化中医药知识图谱与大语言模型深度耦合，实现可溯源、可解释的智能问答，并自动生成小红书风格的科普图文内容，一键发布至社交媒体平台。

### 1.2 项目目标

| 目标 | 说明 |
| --- | --- |
| **知识构建** | 从中医药文献、医案、药典等原始文本中自动抽取实体与关系，构建高质量中医药知识图谱 |
| **智能问答** | 基于知识图谱与向量检索双引擎，实现精准的中医药智能问答，杜绝“幻觉” |
| **内容生成** | 自动将知识图谱中的中医药知识转化为小红书风格图文内容，辅助中医药科普传播 |

---

## 2. 技术架构

### 2.1 总体架构图

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        前端展示层 (Streamlit)                            │
│  ┌───────────┐ ┌───────────────┐ ┌───────────┐ ┌─────────────────────┐ │
│  │ 智能问答   │ │知识图谱可视化  │ │ 图文生成   │ │ 一键发布            │ │
│  └───────────┘ └───────────────┘ └───────────┘ └─────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│                   多智能体协同层 (LangGraph)                             │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│  │ 问答Agent    │ │ 图谱查询Agent│ │ 内容生成Agent │ │ 发布Agent       │ │
│  └─────────────┘ └─────────────┘ └──────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│                   检索增强生成层 (RAG Pipeline)                          │
│  ┌───────────────────────────┐ ┌──────────────────────────────────┐   │
│  │  知识图谱检索 (Neo4j)      │ │   向量相似度检索 (FAISS)          │   │
│  │  Cypher查询 + 图遍历       │ │   GTE Embedding + FAISS索引      │   │
│  └───────────────────────────┘ └──────────────────────────────────┘   │
│                      ↓ 混合检索结果融合 (RRF) ↓                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │          重排序 (达摩院 COROM Rerank) + 上下文构建               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                实体与关系抽取层 (DeepSeek + LangChain)                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  原始文本 → 实体识别 → 关系抽取 → 知识图谱构建                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│                        数据采集与预处理层                                │
│  ┌─────────────────────┐ ┌─────────────────────────────────────┐      │
│  │  爬虫采集            │ │  文本清洗与结构化                     │      │
│  │  (中医药文献/医案)    │ │  (OCR/PDF解析/格式标准化)            │      │
│  └─────────────────────┘ └─────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流转

1. **采集阶段**：爬虫从中医药数据库、古籍文献、临床医案等来源采集原始文本数据。
2. **抽取阶段**：DeepSeek 大模型结合 LangChain 的 `create_extraction_chain` 进行实体与关系抽取，生成结构化三元组。
3. **存储阶段**：三元组存入 Neo4j 图数据库；实体文本经**达摩院 GTE 中文 Embedding** 向量化后存入 FAISS 向量索引。
4. **检索阶段**：用户问题同时触发图谱 Cypher 查询和 FAISS 向量检索，结果经融合排序后返回。
5. **生成阶段**：多 Agent 协同完成答案生成、图文内容创作与平台发布。

---

## 3. 技术栈详解

### 3.1 核心技术栈

| 层级 | 技术选型 | 用途说明 |
| --- | --- | --- |
| 前端界面 | Streamlit | 快速构建交互式问答界面与知识图谱可视化 |
| 多智能体编排 | LangGraph | 基于图结构的多 Agent 协作流程编排，支持条件路由与循环 |
| LLM 接入 | DeepSeek API | 实体关系抽取、问答生成、内容创作的核心大模型 |
| AI 框架 | LangChain | RAG 流水线编排，Cypher 查询生成（`GraphCypherQAChain`） |
| 图数据库 | Neo4j | 中医药知识图谱存储，Cypher 查询支持 |
| 向量检索 | FAISS | 语义向量相似度检索，**达摩院 GTE 中文 Embedding** |
| 重排序 | 达摩院 COROM Rerank | 混合检索结果的精排，提升召回相关性 |
| 模型下载 | ModelScope 魔搭社区 | 国内直连下载国产模型，规避 HuggingFace 网络问题 |
| 数据爬取 | Scrapy / Requests | 中医药文献与医案数据采集 |
| 文档解析 | PyPDF2 / Tesseract OCR | 古籍医案 PDF 与扫描件解析 |
| 可视化 | ECharts / Pyvis | 知识图谱力导向图可视化 |

### 3.2 关键技术选型理由

| 组件 | 选型理由 |
| --- | --- |
| **Neo4j + FAISS 双引擎** | Neo4j 擅长图遍历与关系推理，FAISS 擅长语义相似度匹配，二者互补实现混合检索，相比纯向量检索可显著降低“幻觉”率 |
| **LangGraph** | 相比 LangChain 的链式调用，LangGraph 原生支持分支、循环与并行执行，更适合多 Agent 协作场景 |
| **DeepSeek** | 高性价比的国产大模型，API 形式调用无需本地下载，在中医药领域实体抽取任务上表现优异 |
| **ModelScope 下载国产模型** | 魔搭社区为国内模型托管平台，下载速度快、无需科学上网，保证向量/排序模型可稳定获取 |

---

## 4. 模型国产化与易下载说明

> ⭐ **本节为相对原方案的核心调整**：将原本需从 HuggingFace 下载的、国内难以稳定获取的模型（如 `BAAI/bge-m3`、`BGE-Reranker` 等），统一替换为**国产且可通过 ModelScope 魔搭社区国内直连下载**的模型，保证开箱即用。

### 4.1 替换对照表

| 用途 | 原方案（HuggingFace，难下载） | 替换方案（国产，易下载） | 下载来源 |
| --- | --- | --- | --- |
| 中文文本向量化 | `BAAI/bge-m3` / `bge-large-zh` | **达摩院 GTE**：`damo/nlp_gte_sentence-embedding_chinese-base` | ModelScope 魔搭社区 |
| 检索结果重排序 | `BAAI/bge-reranker` | **达摩院 COROM**：`damo/nlp_corom_ranking_chinese-base` | ModelScope 魔搭社区 |
| 核心大语言模型 | （通用国外 LLM） | **DeepSeek API**（无需本地下载） | DeepSeek 开放平台 |

### 4.2 选型理由

- **达摩院 GTE（General Text Embedding）**：阿里巴巴达摩院出品的国产中文语义向量模型，`base` 版本输出 **768 维**向量，与原方案维度一致，可无缝对接 FAISS；在中文语义检索榜单上表现优异。
- **达摩院 COROM 排序模型**：国产中文精排模型，用于对多路召回结果重排序，与 GTE 同源，生态一致。
- **ModelScope 魔搭社区**：阿里推出的国内模型托管平台，`pip install modelscope` 后即可国内高速下载，首次调用自动拉取模型权重，**彻底规避 HuggingFace 在国内的下载失败/超时问题**。
- **DeepSeek**：国产大模型，以 API 方式接入，本地零下载。

### 4.3 一键安装与模型拉取

```bash
# 安装 ModelScope SDK（国内源安装）
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# 首次运行代码时会自动从魔搭社区下载模型权重，也可手动预下载：
modelscope download --model damo/nlp_gte_sentence-embedding_chinese-base
modelscope download --model damo/nlp_corom_ranking_chinese-base
```

> 💡 若个别环境仍希望使用 HuggingFace 镜像，可设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`，但本项目默认推荐 ModelScope 路线。

---

## 5. 核心功能与实现方案

### 5.1 数据采集与原始文本获取

**功能描述**：从中医药文献、古籍、临床医案、药典等数据源自动采集原始文本数据。

**实现方案**：
- 基于 Scrapy 框架构建定向爬虫，采集公开中医药数据库与文献网站；
- 支持 PDF、Word、TXT 等多格式文件上传解析；
- 对扫描件采用 Tesseract OCR 进行文字识别；
- 采集数据统一存储为结构化 JSON 格式，包含来源、章节、内容等元信息。

### 5.2 实体与关系抽取（DeepSeek + LangChain）

**功能描述**：利用大模型从原始文本中自动抽取中医药实体及实体间关系，构建知识图谱的基础三元组。

**实现方案**：
- 基于 LangChain 的 `create_extraction_chain` 定义实体抽取 Schema，包含药材、方剂、症状、证型、功效等 10+ 实体类型；
- 利用 DeepSeek API 进行实体识别与关系抽取，Prompt 工程引导模型输出标准化 JSON 格式；
- 关系类型定义：包含“主治”“组成”“功效”“禁忌”“归经”等中医药特有语义关系；
- 对抽取结果进行人工校验与规则过滤，确保知识质量。

```python
# 实体抽取示例（LangChain + DeepSeek）
import os
from langchain.chains import create_extraction_chain
from langchain_openai import ChatOpenAI

schema = {
    "properties": {
        "entity_type": {"type": "string",
                        "enum": ["药材", "方剂", "症状", "证型", "功效"]},
        "entity_name": {"type": "string"},
        "relation": {"type": "string",
                     "enum": ["主治", "组成", "功效", "禁忌"]},
        "target": {"type": "string"}
    }
}

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
chain = create_extraction_chain(llm, schema)
result = chain.run("柴胡疏肝散由柴胡、香附、枳壳、陈皮、川芎、白芍、甘草组成，主治肝气郁滞证")
# 输出: [{"entity_type":"方剂","entity_name":"柴胡疏肝散","relation":"组成","target":"柴胡"}, ...]
```

### 5.3 知识图谱构建与存储（Neo4j）

**功能描述**：将抽取的三元组存入 Neo4j 图数据库，构建可查询的中医药知识图谱。

**实现方案**：
- 使用 `py2neo` / `neo4j` 驱动批量写入 Neo4j，节点包含实体属性，关系包含关系类型与置信度；
- 图谱结构设计：药材、方剂、症状、证型、功效、疾病作为主要节点类型；
- 关系类型定义：
  - `(方剂)-[:COMPOSED_OF]->(药材)`：方剂组成
  - `(方剂)-[:TREATS]->(证型)`：方剂主治证型
  - `(证型)-[:HAS_SYMPTOM]->(症状)`：证型对应症状
  - `(药材)-[:HAS_EFFECT]->(功效)`：药材功效
  - `(药材)-[:CONTRAINDICATED_WITH]->(药材)`：配伍禁忌
- 建立索引与约束，优化查询性能。

```cypher
// Neo4j 图谱示例节点与关系
CREATE (f:Fangji {name: "柴胡疏肝散", source: "《景岳全书》"})
CREATE (h:Herb {name: "柴胡"})
CREATE (h2:Herb {name: "香附"})
CREATE (s:Syndrome {name: "肝郁气滞证"})
CREATE (f)-[:COMPOSED_OF {dosage: "12g"}]->(h)
CREATE (f)-[:COMPOSED_OF {dosage: "15g"}]->(h2)
CREATE (f)-[:TREATS]->(s)
```

### 5.4 向量化与索引（FAISS + 达摩院 GTE）

**功能描述**：对实体文本和知识片段进行向量化，构建语义向量检索索引。

**实现方案**：
- 采用**达摩院 GTE 中文模型**（`damo/nlp_gte_sentence-embedding_chinese-base`，经 ModelScope 下载）进行文本向量化，输出 **768 维**向量；
- 对药材功效、方剂主治、症状描述等知识片段分别生成向量；
- 使用 FAISS 构建向量索引，索引类型选用 `IndexFlatIP`（内积相似度）或 `IndexIVFFlat`（大规模数据优化）；
- 向量与 Neo4j 节点 ID 关联映射，支持向量检索后回查图数据库获取完整上下文。

```python
# 基于 ModelScope 的国产 GTE 向量化（国内直连，无需翻墙）
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

embedder = pipeline(
    Tasks.sentence_embedding,
    model="damo/nlp_gte_sentence-embedding_chinese-base",  # 输出 768 维
)

def get_embedding(text: str):
    result = embedder({"source_sentence": [text]})
    return result["text_embedding"][0]
```

```python
# 构建 FAISS 索引
import faiss
import numpy as np

DIM = 768
index = faiss.IndexFlatIP(DIM)           # 内积相似度
# 向量归一化后，内积等价于余弦相似度
vectors = np.array([get_embedding(t) for t in corpus], dtype="float32")
faiss.normalize_L2(vectors)
index.add(vectors)
faiss.write_index(index, "index/tcm_herb.faiss")
```

### 5.5 双引擎混合检索策略

**功能描述**：融合 Neo4j 图查询与 FAISS 向量检索，实现精准的中医药知识召回。

**实现方案**：

| 检索方式 | 权重 | 适用场景 |
| --- | --- | --- |
| Cypher 图查询（Neo4j） | 40% | 关系明确的精确查询（如“柴胡疏肝散的组成”） |
| 向量相似度检索（FAISS + GTE） | 30% | 语义模糊匹配（如“胸闷胁痛用什么方子”） |
| 关键词 BM25 检索 | 30% | 精确术语匹配 |

- 用户问题同时触发三路检索，结果经 **RRF（Reciprocal Rank Fusion）** 融合排序；
- 利用 LangChain 的 `GraphCypherQAChain` 实现自然语言到 Cypher 查询的自动转换；
- 融合后经**达摩院 COROM Rerank** 精排，输出 Top-K 上下文。

```python
# 混合检索示例
from modelscope.pipelines import pipeline as ms_pipeline
from modelscope.utils.constant import Tasks

reranker = ms_pipeline(Tasks.text_ranking,
                       model="damo/nlp_corom_ranking_chinese-base")

def hybrid_search(query: str, top_k: int = 10):
    # 1. 向量检索（GTE + FAISS）
    query_vec = get_embedding(query)
    faiss_results = faiss_index.search(query_vec, top_k * 2)

    # 2. 图谱查询（LangChain 自动生成 Cypher）
    cypher = graph_qa_chain.generate_cypher(query)
    graph_results = neo4j_driver.run(cypher)

    # 3. 关键词检索
    keyword_results = elasticsearch.search(query)

    # 4. RRF 融合排序
    merged = rrf_fusion(faiss_results, graph_results, keyword_results)

    # 5. 达摩院 COROM 精排
    reranked = reranker({"source_sentence": [query],
                         "sentence_to_sort": [m.text for m in merged]})
    return reranked["scores"][:top_k]
```

### 5.6 多智能体协同系统（LangGraph）

**功能描述**：基于 LangGraph 构建多 Agent 协作系统，实现意图识别、任务路由、多专家协同处理。

**Agent 定义**：

| Agent 名称 | 职责 |
| --- | --- |
| 意图识别 Agent | 分析用户问题类型（问答、图谱查询、内容生成、发布） |
| 问答 Agent | 基于 RAG 流水线生成中医药知识问答回答 |
| 图谱查询 Agent | 执行 Cypher 查询，返回结构化图谱数据 |
| 内容生成 Agent | 调用 DeepSeek 生成小红书风格图文内容 |
| 发布 Agent | 对接社交媒体 API，一键发布内容 |

**LangGraph 工作流编排**：

```python
from langgraph.graph import StateGraph, MessagesState

workflow = StateGraph(MessagesState)

# 添加节点
workflow.add_node("intent_agent", identify_intent)
workflow.add_node("qa_agent", qa_handler)
workflow.add_node("graph_agent", graph_query_handler)
workflow.add_node("content_agent", content_generator)
workflow.add_node("publish_agent", publish_handler)

# 条件路由
def router(state):
    intent = state["intent"]
    if intent == "qa":
        return "qa_agent"
    elif intent == "graph_query":
        return "graph_agent"
    elif intent == "content_generation":
        return "content_agent"
    elif intent == "publish":
        return "publish_agent"

workflow.add_conditional_edges("intent_agent", router)
workflow.set_entry_point("intent_agent")
app = workflow.compile()
```

### 5.7 小红书风格图文生成与一键发布

**功能描述**：自动将中医药知识转化为小红书平台适配的图文内容，并一键发布。

**实现方案**：
- 内容生成 Agent 基于 DeepSeek 生成小红书风格文案（Emoji 表情、口语化表达、热门话题标签）；
- 集成 Pillow 生成配图（知识图谱节点截图 + 文字排版）；
- 通过社交媒体开放 API（如小红书开放平台）实现一键发布；
- 支持内容预览、编辑与定时发布功能。

---

## 6. 数据库设计

### 6.1 Neo4j 图模型

| 节点类型 | 主要属性 | 索引字段 |
| --- | --- | --- |
| 药材 Herb | name, property, source | name |
| 方剂 Fangji | name, source, formula | name |
| 症状 Symptom | name, description | name |
| 证型 Syndrome | name, description | name |
| 功效 Effect | name, description | name |
| 疾病 Disease | name, description | name |

### 6.2 FAISS 向量索引

| 索引名称 | 向量维度 | 数据量 | 关联映射 |
| --- | --- | --- | --- |
| herb_embeddings | 768 | N 药材数 | herb_id → Neo4j ID |
| formula_embeddings | 768 | N 方剂数 | formula_id → Neo4j ID |
| symptom_embeddings | 768 | N 症状数 | symptom_id → Neo4j ID |

> 向量均由**达摩院 GTE（768 维）**生成，通过 ModelScope 获取模型。

---

## 7. 项目结构

```text
tcm-knowledge-graph/
├── data/                           # 数据采集与处理
│   ├── crawler/                    # 爬虫模块 (Scrapy)
│   │   ├── spiders/
│   │   └── items.py
│   ├── parser/                     # 文档解析
│   │   ├── pdf_parser.py
│   │   └── ocr_parser.py
│   └── raw_data/                   # 原始数据存储
├── kg/                             # 知识图谱构建
│   ├── entity_extraction/          # 实体抽取 (DeepSeek + LangChain)
│   │   ├── extractor.py
│   │   └── prompt_templates.py
│   ├── relation_extraction/        # 关系抽取
│   ├── neo4j_loader/               # Neo4j 批量写入
│   └── schema/                     # 图谱 Schema 定义
├── retrieval/                      # 检索系统
│   ├── vector_index/               # FAISS 向量索引 (达摩院 GTE)
│   │   ├── index_builder.py
│   │   └── retriever.py
│   ├── graph_retriever/            # Neo4j 图查询
│   │   └── cypher_retriever.py
│   └── hybrid_retriever/           # 混合检索融合 (RRF + COROM Rerank)
│       └── fusion.py
├── agents/                         # 多智能体系统 (LangGraph)
│   ├── intent_agent.py
│   ├── qa_agent.py
│   ├── graph_agent.py
│   ├── content_agent.py
│   └── publish_agent.py
├── web/                            # Streamlit 前端
│   ├── app.py
│   ├── pages/
│   │   ├── qa.py
│   │   ├── graph_viz.py
│   │   └── publish.py
│   └── components/
├── deploy/                         # 部署配置
│   └── docker-compose.yml
├── common/                         # 公共基础模块（配置加载/LLM工厂/GTE向量服务）
├── configs/                        # 配置文件
│   ├── neo4j_config.yaml
│   ├── faiss_config.yaml
│   └── agent_config.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 8. 部署指南

### 8.1 环境要求

| 组件 | 版本要求 |
| --- | --- |
| Python | 3.10+ |
| Neo4j | 4.4+ / AuraDB |
| FAISS | cpu/gpu 版本均可 |
| ModelScope | 最新版（用于下载国产模型） |
| Docker | 20.10+ |
| 内存 | 8GB+（推荐 16GB） |

### 8.2 依赖安装（国内源加速）

```bash
# 使用清华镜像源安装，加速国内下载
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 单独安装 ModelScope（用于拉取达摩院 GTE / COROM 模型）
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**requirements.txt 关键依赖**：

```text
streamlit>=1.30.0
langchain>=0.2.0
langchain-openai>=0.1.0
langgraph>=0.1.0
neo4j>=5.0.0
py2neo>=2021.2.0
faiss-cpu>=1.7.4
modelscope>=1.15.0        # 国产模型下载（达摩院 GTE / COROM）
scrapy>=2.11.0
PyPDF2>=3.0.0
pytesseract>=0.3.10
pillow>=10.0.0
pyvis>=0.3.2
python-dotenv>=1.0.0
```

### 8.3 Docker 一键部署

```bash
# 克隆项目
git clone https://github.com/yourusername/tcm-knowledge-graph.git
cd tcm-knowledge-graph

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DeepSeek API Key、Neo4j 连接信息

# 启动所有服务
docker-compose up -d

# 访问 Streamlit 界面
# http://localhost:8501
```

### 8.4 手动部署步骤

```bash
# 1. 安装 Python 依赖（国内源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 预下载国产向量与排序模型（ModelScope 国内直连）
modelscope download --model damo/nlp_gte_sentence-embedding_chinese-base
modelscope download --model damo/nlp_corom_ranking_chinese-base

# 3. 启动 Neo4j
neo4j start

# 4. 构建 FAISS 索引
python -m retrieval.vector_index.index_builder

# 5. 导入知识图谱数据
python -m kg.neo4j_loader.loader

# 6. 启动 Streamlit 应用
streamlit run web/app.py
```

---

## 9. 性能指标

| 指标 | 目标值 |
| --- | --- |
| 问答响应延迟 | < 3 秒 |
| 向量检索延迟（Top-10） | < 200ms |
| 知识图谱查询延迟 | < 500ms |
| 实体抽取准确率 | > 85% |
| 图文生成耗时 | < 10 秒 |
| 并发支持 | 50 QPS |

---

## 10. 项目路线图

- [x] 数据采集与原始文本获取
- [x] 实体与关系抽取流水线
- [x] Neo4j 知识图谱构建
- [x] FAISS 向量索引（达摩院 GTE 国产化）
- [x] 双引擎混合检索（RRF + COROM Rerank）
- [ ] LangGraph 多 Agent 系统集成
- [ ] Streamlit 前端界面开发
- [ ] 小红书图文生成与发布功能
- [ ] 系统性能优化与压力测试
- [ ] Docker 容器化部署

---

## 附：与原方案的主要差异

| 模块 | 原方案 | 本方案 | 收益 |
| --- | --- | --- | --- |
| 中文向量化 | `BAAI/bge-m3`（HuggingFace） | 达摩院 GTE（ModelScope） | 国内直连、下载快、免翻墙 |
| 结果重排序 | `BGE-Reranker`（HuggingFace） | 达摩院 COROM（ModelScope） | 国产同源、部署稳定 |
| 大语言模型 | 通用 LLM | DeepSeek API | 国产、零本地下载、高性价比 |
| 依赖安装 | PyPI 默认源 | 清华镜像源 | 国内安装提速 |
