"""知识入库编排：文档 → 实体关系抽取 → 校验 → 写入 Neo4j → 重建 FAISS 索引。"""

from kg.entity_extraction.extractor import EntityRelationExtractor
from kg.neo4j_loader import KnowledgeGraphLoader
from kg.relation_extraction.validator import validate_extraction


def ingest_docs(docs: list[dict], rebuild_index: bool = True) -> dict:
    """将结构化文档抽取为三元组并写入知识图谱，最后重建向量索引。

    docs 为 parse_file / 爬虫输出的统一结构：
        {"source": ..., "format": ..., "sections": [{"section": ..., "content": ...}]}

    返回：
        {"docs": [{source, format, nodes, relations, chunks}],
         "nodes", "relations", "chunks", "index"}
    """
    extractor = EntityRelationExtractor()
    loader = KnowledgeGraphLoader()
    try:
        loader.ensure_schema()
        per_doc = []
        seen_nodes, seen_relations = set(), set()
        total_chunks = 0

        for doc in docs:
            doc_nodes, doc_relations, doc_chunks = set(), set(), 0
            for chunk, result in extractor.extract_chunks(doc):
                result = validate_extraction(result)
                loader.load(result, chunk=chunk)
                doc_chunks += 1
                for e in result.entities:
                    doc_nodes.add((e.entity_type, e.entity_name))
                for t in result.triples:
                    doc_relations.add((t.head, t.relation, t.tail))

            seen_nodes.update(doc_nodes)
            seen_relations.update(doc_relations)
            total_chunks += doc_chunks
            per_doc.append({
                "source": doc.get("source", ""),
                "format": doc.get("format", ""),
                "nodes": len(doc_nodes),
                "relations": len(doc_relations),
                "chunks": doc_chunks,
            })

        index_stats = None
        if rebuild_index:
            from retrieval.vector_index.index_builder import rebuild_index as _rebuild
            index_stats = _rebuild()

        return {
            "docs": per_doc,
            "nodes": len(seen_nodes),
            "relations": len(seen_relations),
            "chunks": total_chunks,
            "index": index_stats,
        }
    finally:
        loader.close()
