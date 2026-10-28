"""双引擎混合检索：图谱 + 向量 + BM25 三路召回，RRF 融合，COROM 精排。"""

from common.config import load_config
from retrieval.graph_retriever.cypher_retriever import GraphRetriever
from retrieval.hybrid_retriever.keyword_retriever import KeywordRetriever
from retrieval.vector_index.retriever import VectorRetriever


def rrf_fusion(ranked_lists: list[list[dict]], weights: list[float],
               rrf_k: int = 60) -> list[dict]:
    """RRF（Reciprocal Rank Fusion）加权融合多路排序结果。"""
    fused: dict[str, dict] = {}
    for results, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(results):
            key = (item.get("id") or item.get("chunk_id") or item.get("entity_id")
                   or item.get("name") or item.get("text", "")[:50])
            entry = fused.setdefault(key, {**item, "rrf_score": 0.0, "sources": []})
            entry["rrf_score"] += weight / (rrf_k + rank + 1)
            entry["sources"].append(item.get("retriever", "unknown"))
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


class HybridRetriever:
    """三路混合检索入口，各路独立容错（某一路不可用不影响整体）。"""

    def __init__(self):
        cfg = load_config("agent_config.yaml")
        self.retrieval_cfg = cfg["retrieval"]
        self.rerank_cfg = cfg.get("rerank", {})

        self.vector = VectorRetriever()
        self.keyword = KeywordRetriever()
        try:
            self.graph = GraphRetriever()
        except Exception:
            self.graph = None  # Neo4j 未启动时降级为双路检索
        self._reranker = None

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """执行混合检索，返回精排后的 Top-K 上下文候选。"""
        top_k = top_k or self.retrieval_cfg["top_k"]
        candidate_k = top_k * self.retrieval_cfg.get("candidate_multiplier", 2)
        weights_cfg = self.retrieval_cfg["weights"]

        # 三路并行召回（此处顺序执行，各路独立容错）
        graph_results = self._safe_search(
            self.graph, query, candidate_k, "graph")
        vector_results = self._safe_search(
            self.vector, query, candidate_k, "vector")
        keyword_results = self._safe_search(
            self.keyword, query, candidate_k, "keyword")

        fused = rrf_fusion(
            [graph_results, vector_results, keyword_results],
            [weights_cfg["graph"], weights_cfg["vector"], weights_cfg["keyword"]],
            rrf_k=self.retrieval_cfg.get("rrf_k", 60),
        )

        if self.rerank_cfg.get("enabled"):
            fused = self._rerank(query, fused, top_k * 2)
        return fused[:top_k]

    def build_context(self, results: list[dict]) -> str:
        """将检索结果拼接为供 LLM 使用的参考上下文，原文块附带来源。"""
        lines = []
        for i, item in enumerate(results, 1):
            text = item.get("text") or item.get("name", "")
            if item.get("kind") == "chunk":
                loc = " · ".join(
                    x for x in (item.get("source"), item.get("section")) if x
                )
                prefix = f"（{loc}）" if loc else ""
                lines.append(f"[{i}] {prefix}{text}")
            else:
                source = item.get("source")
                suffix = f"（来源：{source}）" if source else ""
                lines.append(f"[{i}] {text}{suffix}")
        return "\n".join(lines)

    # ---------- 内部方法 ----------

    @staticmethod
    def _safe_search(retriever, query: str, k: int, source: str) -> list[dict]:
        if retriever is None or getattr(retriever, "available", True) is False:
            return []
        try:
            results = retriever.search(query, top_k=k)
            for r in results:
                r["retriever"] = source
            return results
        except Exception:
            return []

    def _rerank(self, query: str, candidates: list[dict], limit: int) -> list[dict]:
        """达摩院 COROM 精排（ModelScope 下载），失败时降级为 RRF 顺序。"""
        texts = [c.get("text") or c.get("name", "") for c in candidates]
        if not texts:
            return candidates
        try:
            if self._reranker is None:
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
                self._reranker = pipeline(
                    Tasks.text_ranking, model=self.rerank_cfg["model"])
            input_data = {"source_sentence": [query], "sentences_to_compare": texts}
            result = self._reranker(input_data)
            order = result.get("index", list(range(len(candidates))))
            scores = result.get("scores")
            reranked = [candidates[i] for i in order if i < len(candidates)]
            if scores:
                for item, score in zip(reranked, scores):
                    item["rerank_score"] = float(score)
            return reranked[:limit]
        except Exception:
            return candidates[:limit]
