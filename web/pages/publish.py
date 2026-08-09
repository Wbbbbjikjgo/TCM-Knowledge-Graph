"""图文生成与发布页面：小红书风格创作 → 预览编辑 → 一键发布。"""

from langchain_core.messages import HumanMessage

import streamlit as st

from agents.content_agent import content_generator, generate_card
from agents.publish_agent import publish_handler

st.header("📕 小红书图文生成与一键发布")
st.caption("基于知识图谱溯源生成科普笔记，支持预览、编辑与发布")

topic = st.text_input(
    "创作主题", value="柴胡疏肝散的养生妙用", placeholder="输入中医药主题，如：艾灸足三里的功效")

if st.button("✨ 生成图文内容", type="primary") and topic.strip():
    with st.spinner("正在基于知识图谱创作（DeepSeek 生成中）..."):
        result = content_generator({"messages": [HumanMessage(content=topic)]})
    st.session_state.draft = result["generated_content"]
    st.success("生成完成，可在下方编辑后发布")

if "draft" in st.session_state:
    st.subheader("内容预览与编辑")
    st.session_state.draft = st.text_area(
        "笔记内容", value=st.session_state.draft, height=300)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🖼️ 生成配图"):
            title = st.session_state.draft.split("\n", 1)[0].replace("【标题】", "").strip()
            card_path = generate_card(title or "中医药小知识", st.session_state.draft)
            st.session_state.card_path = str(card_path)
        if st.session_state.get("card_path"):
            st.image(st.session_state.card_path, caption="笔记配图")

    with col2:
        if st.button("🚀 一键发布到小红书"):
            result = publish_handler({"generated_content": st.session_state.draft})
            if result.get("publish_result", {}).get("success"):
                st.success(result["answer"])
            else:
                st.warning(result["answer"])
