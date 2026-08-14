"""数据导入页面：文件上传解析 + 网页采集 → 抽取 → 写入知识图谱 → 重建向量索引。"""

import tempfile
from pathlib import Path

import streamlit as st

from data.crawler import crawl_urls
from data.parser import parse_file, save_parsed

st.header("📥 数据导入")
st.caption("上传文档或抓取网页 → 解析 → DeepSeek 抽取实体关系 → 写入 Neo4j → 重建 FAISS 索引")

tab_file, tab_crawl, tab_index = st.tabs(["📄 文件上传", "🕷️ 网页采集", "🔧 索引维护"])


def _render_stats(stats: dict) -> None:
    st.success(f"入库完成：{stats['nodes']} 个节点、{stats['relations']} 条关系")
    if stats.get("index"):
        idx_text = "、".join(f"{k}: {v} 条" for k, v in stats["index"].items())
        st.info(f"FAISS 向量索引已重建：{idx_text}")
    if stats.get("docs"):
        st.dataframe([
            {"来源": d["source"], "格式": d.get("format", ""),
             "节点数": d["nodes"], "关系数": d["relations"]}
            for d in stats["docs"]
        ])


def _ingest(docs: list[dict]) -> None:
    from kg.ingest import ingest_docs
    with st.spinner("DeepSeek 抽取实体关系并写入知识图谱（可能需要较长时间）..."):
        stats = ingest_docs(docs)
    _render_stats(stats)


# ---------- 文件上传 ----------
with tab_file:
    st.subheader("上传文档（PDF / TXT / MD / 图片）")
    uploaded_files = st.file_uploader(
        "选择文件", type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if st.button("解析并入库", type="primary", key="file_ingest"):
        if not uploaded_files:
            st.warning("请先上传文件")
        else:
            docs = []
            parse_info = []
            with st.spinner("解析文件中..."):
                for uf in uploaded_files:
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp_path = Path(tmp) / uf.name
                        tmp_path.write_bytes(uf.getbuffer())
                        doc = parse_file(tmp_path)
                        saved = save_parsed(doc)
                        docs.append(doc)
                        parse_info.append(f"{uf.name} → 已保存 {saved.name}")
            for line in parse_info:
                st.caption(line)
            try:
                _ingest(docs)
            except Exception as e:
                st.error(f"入库失败：{e}")

# ---------- 网页采集 ----------
with tab_crawl:
    st.subheader("网页采集（填写目标链接）")
    st.caption("逐行填写要抓取的页面链接，抓取正文后抽取实体关系并入库")
    urls_text = st.text_area(
        "目标链接（每行一个）",
        placeholder="https://www.example.com/article/1\nhttps://www.example.com/article/2",
        height=120,
    )
    if st.button("抓取并入库", type="primary", key="crawl_ingest"):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not urls:
            st.warning("请先填写链接")
        else:
            with st.spinner("抓取网页中（1 秒礼貌间隔，遵守 robots.txt）..."):
                result = crawl_urls(urls)

            if result["error"]:
                st.error(f"抓取失败：{result['error']}")
            if result["blocked"]:
                st.warning("以下链接未抓取：")
                for b in result["blocked"]:
                    st.markdown(f"- {b['url']}：{b['reason']}")
            if result["items"]:
                st.success(f"成功抓取 {len(result['items'])} 篇内容")
                docs = [{
                    "source": it["source"],
                    "format": "html",
                    "sections": [{"section": it.get("title") or "正文",
                                  "content": it.get("content", "")}],
                } for it in result["items"]]
                try:
                    _ingest(docs)
                except Exception as e:
                    st.error(f"入库失败：{e}")
            else:
                st.info(
                    "未抓取到任何正文内容。可能原因：页面结构不匹配、"
                    "robots.txt 禁止或网络错误。"
                )

# ---------- 索引维护 ----------
with tab_index:
    st.subheader("向量索引维护")
    st.caption("知识图谱变更后需重建 FAISS 索引，向量检索才能命中新数据")
    if st.button("重建 FAISS 向量索引", type="primary"):
        try:
            from retrieval.vector_index.index_builder import rebuild_index
            with st.spinner("从 Neo4j 读取实体并重建索引..."):
                stats = rebuild_index()
            idx_text = "、".join(f"{k}: {v} 条" for k, v in stats.items())
            st.success(f"索引重建完成：{idx_text}")
        except Exception as e:
            st.error(f"索引重建失败，请确认 Neo4j 已启动且已导入数据：{e}")
