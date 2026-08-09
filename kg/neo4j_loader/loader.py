"""Neo4j 知识图谱批量写入：索引构建、节点/关系 MERGE、CLI 入口。"""

import argparse
import os

from neo4j import GraphDatabase

from common.config import load_config
from kg.entity_extraction.extractor import Entity, ExtractionResult, Triple
from kg.schema import ENTITY_LABEL_MAP, RELATION_TYPE_MAP


def get_driver():
    """创建 Neo4j 驱动（连接信息来自 .env）。"""
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )


class KnowledgeGraphLoader:
    """将抽取结果写入 Neo4j，构建中医药知识图谱。"""

    def __init__(self, driver=None):
        self.driver = driver or get_driver()

    def ensure_schema(self):
        """按配置为各节点标签的 name 字段创建索引。"""
        cfg = load_config("neo4j_config.yaml")["schema"]
        with self.driver.session() as session:
            for node in cfg["nodes"]:
                session.run(
                    f"CREATE INDEX {node['label'].lower()}_{node['index_field']}_idx IF NOT EXISTS "
                    f"FOR (n:{node['label']}) ON (n.{node['index_field']})"
                )

    def load(self, result: ExtractionResult) -> dict:
        """批量写入实体节点与关系三元组，返回写入统计。"""
        entity_type_of = {e.entity_name: e.entity_type for e in result.entities}
        node_count = self._merge_nodes(result.entities)
        rel_count = self._merge_relations(result.triples, entity_type_of)
        return {"nodes": node_count, "relations": rel_count}

    def _merge_nodes(self, entities: list[Entity]) -> int:
        by_label: dict[str, list[str]] = {}
        for e in entities:
            label = ENTITY_LABEL_MAP.get(e.entity_type)
            if label:
                by_label.setdefault(label, []).append(e.entity_name)

        count = 0
        with self.driver.session() as session:
            for label, names in by_label.items():
                # 标签来自受控 Schema 映射，无注入风险
                session.run(
                    f"UNWIND $names AS name MERGE (n:{label} {{name: name}})",
                    names=list(dict.fromkeys(names)),
                )
                count += len(set(names))
        return count

    def _merge_relations(self, triples: list[Triple], entity_type_of: dict) -> int:
        count = 0
        with self.driver.session() as session:
            for t in triples:
                head_label = ENTITY_LABEL_MAP.get(entity_type_of.get(t.head, ""))
                tail_label = ENTITY_LABEL_MAP.get(entity_type_of.get(t.tail, ""))
                rel_type = RELATION_TYPE_MAP.get(t.relation)
                if not (head_label and tail_label and rel_type):
                    continue
                session.run(
                    f"MATCH (a:{head_label} {{name: $head}}) "
                    f"MATCH (b:{tail_label} {{name: $tail}}) "
                    f"MERGE (a)-[:{rel_type}]->(b)",
                    head=t.head,
                    tail=t.tail,
                )
                count += 1
        return count

    def close(self):
        self.driver.close()


def _demo_result() -> ExtractionResult:
    """演示数据：柴胡疏肝散示例三元组。"""
    entities = [
        Entity(entity_type="方剂", entity_name="柴胡疏肝散"),
        Entity(entity_type="药材", entity_name="柴胡"),
        Entity(entity_type="药材", entity_name="香附"),
        Entity(entity_type="证型", entity_name="肝郁气滞证"),
    ]
    triples = [
        Triple(head="柴胡疏肝散", relation="组成", tail="柴胡"),
        Triple(head="柴胡疏肝散", relation="组成", tail="香附"),
        Triple(head="柴胡疏肝散", relation="主治", tail="肝郁气滞证"),
    ]
    return ExtractionResult(entities=entities, triples=triples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neo4j 知识图谱导入")
    parser.add_argument("--demo", action="store_true", help="导入柴胡疏肝散示例数据")
    args = parser.parse_args()

    loader = KnowledgeGraphLoader()
    try:
        loader.ensure_schema()
        if args.demo:
            stats = loader.load(_demo_result())
            print(f"演示数据写入完成: {stats}")
        else:
            print("请通过 KnowledgeGraphLoader.load() 传入抽取结果，或使用 --demo")
    finally:
        loader.close()
