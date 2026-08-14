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


def fetch_entities(driver, label: str) -> list[dict]:
    """读取指定标签的全部节点，拼接用于向量化的描述文本。"""
    with driver.session() as session:
        records = session.run(
            f"MATCH (n:{label}) RETURN elementId(n) AS id, n.name AS name, "
            f"coalesce(n.description, '') AS description"
        ).data()
    entities = []
    for r in records:
        text = r["name"] + ("：" + r["description"] if r["description"] else "")
        entities.append({
            "entity_id": r["id"],
            "label": label,
            "name": r["name"],
            "text": text,  # 向量化文本
        })
    return entities


def build_collection(driver, embedder, collection: dict, out_dir: Path) -> int:
    """构建单个集合的 FAISS 索引与 ID 映射文件，返回实体数。"""
    entities = fetch_entities(driver, collection["entity_label"])
    if not entities:
        return 0

    vectors = embedder.encode([e["text"] for e in entities])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))

    faiss.write_index(index, str(out_dir / f"{collection['name']}.faiss"))
    with open(out_dir / f"{collection['name']}_map.json", "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    return len(entities)


def rebuild_index() -> dict:
    """从 Neo4j 读取全部集合实体并重建 FAISS 索引，返回 {集合名: 实体数}。

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
        print(f"[{name}] 索引构建完成，实体数: {n}")


if __name__ == "__main__":
    main()
