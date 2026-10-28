"""智能问答页面：RAG 问答 + 打字机流式输出 + 知识溯源。"""

import streamlit as st

from agents.qa_agent import qa_stream

st.header("💬 中医药智能问答")
st.caption("基于知识图谱 + 向量双引擎混合检索，回答可溯源、拒绝幻觉")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def _source_markdown(index: int, source: dict) -> str:
    is_chunk = source.get("kind") == "chunk"
    title = (source.get("section") if is_chunk else source.get("name")) \
        or source.get("name") or "原文片段"
    meta = " · ".join(
        x for x in (
            source.get("label") or ("原文" if is_chunk else ""),
            source.get("source"),
        ) if x
    )
    head = f"**[{index}] {title}**"
    if meta:
        head += f"（{meta}）"
    return f"{head}\n{source.get('text', '')}"


for item in st.session_state.chat_history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            with st.expander(f"📚 知识溯源（{len(item['sources'])} 条依据）"):
                for i, s in enumerate(item["sources"], 1):
                    st.markdown(_source_markdown(i, s))

question = st.chat_input("请输入中医药相关问题，如：柴胡疏肝散由哪些药材组成？")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources = []

        def _text_stream():
            global sources
            for event in qa_stream(question):
                if event["type"] == "sources":
                    sources = event["data"]
                else:
                    yield event["data"]

        # 打字机效果：逐块流式渲染回答
        answer = st.write_stream(_text_stream())

        if sources:
            with st.expander(f"📚 知识溯源（{len(sources)} 条依据）"):
                for i, s in enumerate(sources, 1):
                    st.markdown(_source_markdown(i, s))

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
