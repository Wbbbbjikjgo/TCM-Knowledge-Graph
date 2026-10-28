"""知识图谱可视化页面：Pyvis 力导向图展示实体邻域子图。"""

import streamlit as st
import streamlit.components.v1 as components

st.header("🕸️ 知识图谱可视化")
st.caption("输入实体名称，展示其邻域子图（节点颜色按类型区分）")

LABEL_COLORS = {
    "Herb": "#52c41a",      # 药材-绿
    "Fangji": "#1890ff",    # 方剂-蓝
    "Symptom": "#faad14",   # 症状-黄
    "Syndrome": "#f5222d",  # 证型-红
    "Effect": "#722ed1",    # 功效-紫
    "Disease": "#13c2c2",   # 疾病-青
}
LABEL_NAMES = {
    "Herb": "药材", "Fangji": "方剂", "Symptom": "症状",
    "Syndrome": "证型", "Effect": "功效", "Disease": "疾病",
}

entity_name = st.text_input("中心实体名称", value="柴胡疏肝散")

if st.button("查询并渲染", type="primary") and entity_name.strip():
    try:
        from kg.neo4j_loader import get_driver

        driver = get_driver()
        with driver.session() as session:
            records = session.run(
                """
                MATCH (a {name: $name})-[r]-(b)
                WHERE NOT b:Chunk
                RETURN labels(a)[0] AS alabel, a.name AS aname,
                       type(r) AS rel, labels(b)[0] AS blabel, b.name AS bname
                LIMIT 100
                """,
                name=entity_name.strip(),
            ).data()
        driver.close()

        if not records:
            st.warning("图谱中未找到该实体的关联数据，请先导入知识图谱。")
            st.stop()

        from pyvis.network import Network

        net = Network(height="600px", width="100%", directed=True)
        added = set()
        for r in records:
            for label, name in ((r["alabel"], r["aname"]), (r["blabel"], r["bname"])):
                if name not in added:
                    net.add_node(
                        name, label=f"{name}\n({LABEL_NAMES.get(label, label)})",
                        color=LABEL_COLORS.get(label, "#999999"),
                    )
                    added.add(name)
            net.add_edge(r["aname"], r["bname"], title=r["rel"], label=r["rel"])

        html_path = "graph_viz.html"
        net.write_html(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=640, scrolling=True)

        st.success(f"共渲染 {len(added)} 个节点、{len(records)} 条关系")
    except Exception as e:
        st.error(f"图谱查询失败，请确认 Neo4j 已启动：{e}")
