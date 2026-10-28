"""Neo4j 图谱检索器：实体名匹配、邻域事实扩展、自然语言转 Cypher。"""

import re

from common.config import load_config
from common.llm import get_llm
from kg.neo4j_loader import get_driver

CYPHER_PROMPT = """你是 Neo4j Cypher 查询专家。根据图谱 Schema 将用户问题转换为只读 Cypher 查询。

图谱 Schema：
{schema}

要求：
1. 只允许 MATCH / RETURN / WHERE / LIMIT，禁止任何写操作
2. 节点名称用中文精确匹配或 CONTAINS 模糊匹配
3. 只输出 Cypher 语句本身，不要任何解释

用户问题：{question}
"""

# 只读 Cypher 白名单校验
_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+db\.|LOAD)\b", re.I
)


class GraphRetriever:
    """基于 Neo4j 的结构化图谱检索。"""

    def __init__(self, driver=None):
        self.driver = driver or get_driver()
        self.labels = [
            n["label"] for n in load_config("neo4j_config.yaml")["schema"]["nodes"]
        ]

    # ---------- 结构化检索 ----------

    def find_entities(self, query: str, limit: int = 5) -> list[dict]:
        """跨标签查找问题中提及的实体（问题文本包含实体名）。"""
        found = []
        with self.driver.session() as session:
            for label in self.labels:
                records = session.run(
                    f"MATCH (n:{label}) WHERE $q CONTAINS n.name "
                    f"RETURN elementId(n) AS id, n.name AS name LIMIT $limit",
                    q=query,
                    limit=limit,
                ).data()
                found.extend(
                    {"id": r["id"], "name": r["name"], "label": label}
                    for r in records
                )
        return found[:limit]

    def neighborhood_facts(self, name: str, label: str, limit: int = 20) -> list[str]:
        """获取节点一跳邻域的事实描述（可读文本行）。"""
        with self.driver.session() as session:
            records = session.run(
                f"MATCH (a:{label} {{name: $name}})-[r]-(b) "
                f"WHERE type(r) <> 'MENTIONED_IN' "
                f"RETURN type(r) AS rel, labels(b)[0] AS blabel, b.name AS bname, "
                f"startNode(r).name = $name AS outgoing LIMIT $limit",
                name=name,
                limit=limit,
            ).data()
        facts = []
        for r in records:
            if r["outgoing"]:
                facts.append(f"({name})-[:{r['rel']}]->({r['bname']})")
            else:
                facts.append(f"({r['bname']})-[:{r['rel']}]->({name})")
        return facts

    def chunk_evidence(self, name: str, label: str, limit: int = 3) -> list[dict]:
        """获取实体出现的原文片段，用于回答溯源。"""
        with self.driver.session() as session:
            return session.run(
                f"MATCH (a:{label} {{name: $name}})-[:MENTIONED_IN]->(c:Chunk) "
                f"RETURN c.chunk_id AS chunk_id, c.text AS text, "
                f"c.source AS source, c.section AS section, "
                f"c.chunk_seq AS seq "
                f"ORDER BY c.chunk_seq LIMIT $limit",
                name=name,
                limit=limit,
            ).data()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """图谱结构化检索：实体匹配 + 邻域扩展 + 原文证据。"""
        results = []
        for entity in self.find_entities(query, limit=top_k):
            facts = self.neighborhood_facts(entity["name"], entity["label"])
            evidence = self.chunk_evidence(entity["name"], entity["label"])
            if not (facts or evidence):
                continue
            text = f"{entity['name']}：" + ("；".join(facts) if facts else "无关联事实")
            if evidence:
                text += "\n原文：" + "；".join(e["text"] for e in evidence[:2])
            results.append({
                "id": entity["id"],
                "entity_id": entity["id"],
                "kind": "entity",
                "name": entity["name"],
                "label": entity["label"],
                "text": text,
                "facts": facts,
                "evidence": evidence,
                "source": evidence[0]["source"] if evidence else "",
                "section": evidence[0]["section"] if evidence else "",
            })
        return results[:top_k]

    # ---------- 自然语言转 Cypher ----------

    def generate_cypher(self, question: str) -> str:
        """用 DeepSeek 将自然语言转换为 Cypher（含只读白名单校验）。"""
        schema = self._schema_text()
        resp = get_llm(temperature=0).invoke(
            CYPHER_PROMPT.format(schema=schema, question=question)
        )
        cypher = resp.content.strip().strip("`")
        if cypher.upper().startswith("CYPHER"):
            cypher = cypher.split("\n", 1)[-1]
        if _FORBIDDEN.search(cypher):
            raise ValueError(f"生成的 Cypher 含写操作，已拒绝执行: {cypher}")
        return cypher

    def run_cypher(self, cypher: str, limit: int = 50) -> list[dict]:
        """执行只读 Cypher 并返回记录列表。"""
        if _FORBIDDEN.search(cypher):
            raise ValueError("禁止执行非只读 Cypher")
        with self.driver.session() as session:
            return session.run(cypher + f" LIMIT {limit}").data()

    def _schema_text(self) -> str:
        cfg = load_config("neo4j_config.yaml")["schema"]
        nodes = "、".join(n["label"] for n in cfg["nodes"])
        rels = "、".join(cfg["relations"])
        return (
            f"节点标签: {nodes}（均有 name 属性）\n"
            f"关系类型: {rels}\n"
            f"示例: MATCH (f:Fangji {{name: '柴胡疏肝散'}})-[:COMPOSED_OF]->(h:Herb) RETURN h.name"
        )
