"""知识入库编排：文档 → 实体关系抽取 → 校验 → 写入 Neo4j → 重建 FAISS 索引。"""

from kg.entity_extraction.extractor import EntityRelationExtractor
from kg.neo4j_loader import KnowledgeGraphLoader
from kg.relation_extraction.validator import validate_extraction


def ingest_docs(docs: list[dict], rebuild_index: bool = True) -> dict:
    """将结构化文档抽取为三元组并写入知识图谱，最后重建向量索引。

    docs 为 parse_file / 爬虫输出的统一结构：
        {"source": ..., "format": ..., "sections": [{"section": ..., "content": ...}]}

    返回：
        {"docs": [{source, format, nodes, relations}], "nodes", "relations", "index"}
    """
    extractor = EntityRelationExtractor()
    loader = KnowledgeGraphLoader()
    try:
        loader.ensure_schema()
        per_doc = []
        total_nodes = total_relations = 0
        for doc in docs:
            result = extractor.extract_document(doc)
            result = validate_extraction(result)
            stats = loader.load(result)
            total_nodes += stats["nodes"]
            total_relations += stats["relations"]
            per_doc.append({
                "source": doc.get("source", ""),
                "format": doc.get("format", ""),
                "nodes": stats["nodes"],
                "relations": stats["relations"],
            })

        index_stats = None
        if rebuild_index:
            from retrieval.vector_index.index_builder import rebuild_index as _rebuild
            index_stats = _rebuild()

        return {
            "docs": per_doc,
            "nodes": total_nodes,
            "relations": total_relations,
            "index": index_stats,
        }
    finally:
        loader.close()
