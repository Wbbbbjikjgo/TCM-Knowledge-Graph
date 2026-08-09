"""问答 Agent：RAG 流水线（混合检索 → 上下文构建 → DeepSeek 生成）。"""

from common.llm import get_llm
from retrieval.hybrid_retriever import HybridRetriever

QA_PROMPT = """你是严谨的中医药领域专家助手。请仅依据下方【参考资料】回答用户问题。

要求：
1. 回答必须有据可依，禁止编造资料中不存在的药材、方剂或功效
2. 若参考资料不足以回答，明确告知用户"知识库中暂无相关记载"
3. 在回答末尾标注引用的资料编号，如 [1][3]
4. 涉及用药的内容需提醒"请遵医嘱"

【参考资料】
{context}

【用户问题】
{question}"""

_hybrid: HybridRetriever | None = None


def _get_hybrid() -> HybridRetriever:
    global _hybrid
    if _hybrid is None:
        _hybrid = HybridRetriever()
    return _hybrid


def qa_handler(state: dict) -> dict:
    """LangGraph 节点：执行 RAG 问答，返回答案与溯源信息。"""
    question = state["messages"][-1].content
    hybrid = _get_hybrid()

    results = hybrid.search(question)
    context = hybrid.build_context(results)

    answer = get_llm().invoke(
        QA_PROMPT.format(context=context or "（无相关资料）", question=question)
    ).content

    sources = [
        {"name": r.get("name", ""), "label": r.get("label", ""),
         "text": r.get("text", ""), "retrieved_by": r.get("sources", [])}
        for r in results
    ]
    return {"answer": answer, "sources": sources}


def qa_stream(question: str):
    """流式问答生成器（供前端打字机效果使用），先 yield 溯源再 yield 文本块。"""
    hybrid = _get_hybrid()
    results = hybrid.search(question)
    context = hybrid.build_context(results)

    yield {"type": "sources", "data": results}

    stream = get_llm().stream(
        QA_PROMPT.format(context=context or "（无相关资料）", question=question)
    )
    for chunk in stream:
        if chunk.content:
            yield {"type": "text", "data": chunk.content}
