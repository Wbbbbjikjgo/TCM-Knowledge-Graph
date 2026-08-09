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
                "index": faiss.read_index(str(faiss_path)),
                "entity_map": entity_map,
            })

    @property
    def available(self) -> bool:
        return len(self.collections) > 0

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """返回按相似度降序的候选：{entity_id, label, name, text, score}"""
        if not self.available:
            return []

        query_vec = EmbeddingService.get_instance().encode([query])
        query_vec = np.ascontiguousarray(query_vec)

        results = []
        for collection in self.collections:
            index, entity_map = collection["index"], collection["entity_map"]
            k = min(top_k, index.ntotal)
            if k == 0:
                continue
            scores, idxs = index.search(query_vec, k)
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                entity = entity_map[idx]
                results.append({
                    "entity_id": entity["entity_id"],
                    "label": entity["label"],
                    "name": entity["name"],
                    "text": entity["text"],
                    "score": float(score),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
