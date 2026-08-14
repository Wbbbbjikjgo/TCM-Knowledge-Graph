"""中医药知识图谱智能问答系统 - Streamlit 主入口。"""

import sys
from pathlib import Path

# 保证项目根目录在模块搜索路径中（页面脚本可导入 agents/retrieval 等包）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="中医药知识图谱智能问答",
    page_icon="🌿",
    layout="wide",
)

qa_page = st.Page("pages/qa.py", title="智能问答", icon="💬", default=True)
graph_page = st.Page("pages/graph_viz.py", title="知识图谱可视化", icon="🕸️")
data_page = st.Page("pages/data_import.py", title="数据导入", icon="📥")
publish_page = st.Page("pages/publish.py", title="图文生成与发布", icon="📕")

nav = st.navigation([qa_page, graph_page, data_page, publish_page])

with st.sidebar:
    st.title("🌿 中医药知识图谱")
    st.caption(
        "知识图谱 + RAG 双引擎 · DeepSeek 大模型\n\n"
        "向量模型：达摩院 GTE（ModelScope）\n"
        "重排序：达摩院 COROM"
    )

nav.run()
