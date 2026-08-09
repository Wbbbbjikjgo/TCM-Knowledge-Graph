"""图谱查询 Agent：自然语言转 Cypher，执行只读查询返回结构化数据。"""

from retrieval.graph_retriever import GraphRetriever

_retriever: GraphRetriever | None = None


def _get_retriever() -> GraphRetriever:
    global _retriever
    if _retriever is None:
        _retriever = GraphRetriever()
    return _retriever


def graph_query_handler(state: dict) -> dict:
    """LangGraph 节点：生成并执行 Cypher，返回查询结果。"""
    question = state["messages"][-1].content
    retriever = _get_retriever()

    try:
        cypher = retriever.generate_cypher(question)
        records = retriever.run_cypher(cypher)
        answer = _format_records(question, records)
        return {"answer": answer, "cypher": cypher, "graph_data": records}
    except Exception as e:
        return {"answer": f"图谱查询失败：{e}", "cypher": "", "graph_data": []}


def _format_records(question: str, records: list[dict]) -> str:
    if not records:
        return "图谱中未查询到相关数据。"
    lines = [f"针对「{question}」的图谱查询结果（共 {len(records)} 条）："]
    for i, record in enumerate(records, 1):
        fields = "，".join(f"{k}: {v}" for k, v in record.items())
        lines.append(f"{i}. {fields}")
    return "\n".join(lines)
