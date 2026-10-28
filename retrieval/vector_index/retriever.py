"""FAISS 向量检索器：跨集合语义检索。"""

import json

import faiss
import numpy as np

from common.config import PROJECT_ROOT, load_config
from common.embedding import EmbeddingService


class VectorRetriever:
    """加载全部 FAISS 集合索引，对查询做跨集合语义召回。"""

    def __init__(self):
        cfg = load_config("faiss_config.yaml")
        self.index_dir = PROJECT_ROOT / cfg["index"]["dir"]
        self.collections: list[dict] = []

        for collection in cfg["collections"]:
            faiss_path = self.index_dir / f"{collection['name']}.faiss"
            map_path = self.index_dir / f"{collection['name']}_map.json"
            if not (faiss_path.exists() and map_path.exists()):
                continue  # 索引未构建的集合跳过
            with open(map_path, "r", encoding="utf-8") as f:
                entity_map = json.load(f)
            self.collections.append({
                "name": collection["name"],
                "kind": collection.get("kind", "entity"),
                "index": faiss.read_index(str(faiss_path)),
                "entity_map": entity_map,
            })

    @property
    def available(self) -> bool:
        return len(self.collections) > 0

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """返回按相似度降序的候选，统一携带来源与顺序元数据。"""
        if not self.available:
            return []

        query_vec = EmbeddingService.get_instance().encode([query])
        query_vec = np.ascontiguousarray(query_vec)

        results = []
        for collection in self.collections:
            index, entity_map = collection["index"], collection["entity_map"]
            kind = collection["kind"]
            k = min(top_k, index.ntotal)
            if k == 0:
                continue
            scores, idxs = index.search(query_vec, k)
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                item = entity_map[idx]
                item_id = item.get("id") or item.get("entity_id", "")
                results.append({
                    "id": item_id,
                    "entity_id": item.get("entity_id", ""),
                    "chunk_id": item.get("chunk_id", item_id if kind == "chunk" else ""),
                    "kind": item.get("kind", kind),
                    "label": item.get("label", ""),
                    "name": item.get("name", ""),
                    "text": item.get("text", ""),
                    "source": item.get("source", ""),
                    "section": item.get("section", ""),
                    "order": item.get("order", 0),
                    "score": float(score),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
