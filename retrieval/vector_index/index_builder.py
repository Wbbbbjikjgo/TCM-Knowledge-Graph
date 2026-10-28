"""FAISS 向量索引构建：从 Neo4j 读取实体 → GTE 向量化 → 写入 FAISS。

运行方式: python -m retrieval.vector_index.index_builder
"""

import json
from pathlib import Path

import faiss
import numpy as np

from common.config import PROJECT_ROOT, load_config
from common.embedding import EmbeddingService
from kg.neo4j_loader import get_driver


def fetch_chunks(driver) -> list[dict]:
    """读取全部原文块，向量化文本即原文本身。"""
    with driver.session() as session:
        records = session.run(
            "MATCH (c:Chunk) "
            "RETURN c.chunk_id AS id, c.text AS text, c.source AS source, "
            "c.section AS section, c.chunk_seq AS seq "
            "ORDER BY c.source, c.chunk_seq"
        ).data()

    chunks = []
    for r in records:
        text = (r["text"] or "").strip()
        if not text:
            continue
        chunks.append({
            "id": r["id"],
            "chunk_id": r["id"],
            "kind": "chunk",
            "label": "Chunk",
            "name": r["section"] or r["source"] or "原文片段",
            "text": text,
            "source": r["source"] or "",
            "section": r["section"] or "",
            "order": r["seq"] or 0,
        })
    return chunks


def fetch_entities(driver, label: str) -> list[dict]:
    """读取指定标签节点，文本由名称、描述与其出现的原文片段组成。"""
    with driver.session() as session:
        records = session.run(
            f"MATCH (n:{label}) "
            f"OPTIONAL MATCH (n)-[:MENTIONED_IN]->(c:Chunk) "
            f"WITH n, collect(DISTINCT c.text)[0..3] AS snippets "
            f"RETURN elementId(n) AS id, coalesce(n.name, '') AS name, "
            f"coalesce(n.description, '') AS description, snippets "
            f"ORDER BY name"
        ).data()

    entities = []
    for r in records:
        name = (r["name"] or "").strip()
        if not name:
            continue
        snippets = [s for s in (r["snippets"] or []) if s]
        text = name
        if r["description"]:
            text += "：" + r["description"]
        if snippets:
            text += "。相关原文：" + "；".join(snippets)
        entities.append({
            "id": r["id"],
            "entity_id": r["id"],
            "kind": "entity",
            "label": label,
            "name": name,
            "text": text,
            "source": "",
            "section": "",
            "order": 0,
        })
    return entities


def build_collection(driver, embedder, collection: dict, out_dir: Path) -> int:
    """构建单个集合的 FAISS 索引与映射文件，返回记录数。"""
    if collection.get("kind") == "chunk":
        records = fetch_chunks(driver)
    else:
        records = fetch_entities(driver, collection["entity_label"])
    if not records:
        return 0

    vectors = embedder.encode([e["text"] for e in records])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))

    faiss.write_index(index, str(out_dir / f"{collection['name']}.faiss"))
    with open(out_dir / f"{collection['name']}_map.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return len(records)


def rebuild_index() -> dict:
    """从 Neo4j 读取原文块与实体并重建 FAISS 索引，返回 {集合名: 记录数}。

    供知识入库后（数据导入页面）复用，避免索引与实际图谱不一致。
    """
    cfg = load_config("faiss_config.yaml")
    out_dir = PROJECT_ROOT / cfg["index"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = EmbeddingService.get_instance()
    driver = get_driver()
    stats = {}
    try:
        for collection in cfg["collections"]:
            n = build_collection(driver, embedder, collection, out_dir)
            stats[collection["name"]] = n
    finally:
        driver.close()
    return stats


def main():
    stats = rebuild_index()
    for name, n in stats.items():
        print(f"[{name}] 索引构建完成，记录数: {n}")


if __name__ == "__main__":
    main()
