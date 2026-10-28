"""BM25 关键词检索器：面向中医药术语的精确召回。"""

import json

from rank_bm25 import BM25Okapi

from common.config import PROJECT_ROOT, load_config


def _tokenize(text: str) -> list[str]:
    """中文二元分词 + 连续英文/数字整词，兼顾术语与拉丁名。"""
    tokens, buf = [], ""
    for ch in text:
        if ch.isalnum():
            buf += ch
        else:
            if buf:
                tokens.append(buf.lower())
                buf = ""
    if buf:
        tokens.append(buf.lower())
    # 中文部分做字二元组
    cn = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    tokens += [cn[i:i + 2] for i in range(len(cn) - 1)]
    return tokens


class KeywordRetriever:
    """基于 FAISS 集合实体映射构建 BM25 语料库。"""

    def __init__(self):
        cfg = load_config("faiss_config.yaml")
        index_dir = PROJECT_ROOT / cfg["index"]["dir"]

        self.corpus: list[dict] = []
        for collection in cfg["collections"]:
            map_path = index_dir / f"{collection['name']}_map.json"
            if not map_path.exists():
                continue
            with open(map_path, "r", encoding="utf-8") as f:
                self.corpus.extend(json.load(f))

        tokenized = [_tokenize(e["text"]) for e in self.corpus]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    @property
    def available(self) -> bool:
        return self.bm25 is not None

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.available:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                break
            e = self.corpus[i]
            item_id = e.get("id") or e.get("entity_id", "")
            kind = e.get("kind", "entity")
            results.append({
                "id": item_id,
                "entity_id": e.get("entity_id", ""),
                "chunk_id": e.get("chunk_id", item_id if kind == "chunk" else ""),
                "kind": kind,
                "label": e.get("label", ""),
                "name": e.get("name", ""),
                "text": e.get("text", ""),
                "source": e.get("source", ""),
                "section": e.get("section", ""),
                "order": e.get("order", 0),
                "score": float(scores[i]),
            })
        return results
