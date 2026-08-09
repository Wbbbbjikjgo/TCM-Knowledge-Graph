"""实体与关系抽取 Prompt 模板。"""

EXTRACTION_SYSTEM_PROMPT = """你是中医药领域的知识图谱构建专家。请从给定文本中抽取实体与关系三元组。

规则：
1. 实体类型只能是：{entity_types}
2. 关系类型只能是：{relation_types}
3. 实体名称使用规范名称，去除多余修饰（如"去核山茱萸"记为"山茱萸"需谨慎，保留原文关键信息）
4. 只抽取文本中明确表述的事实，禁止臆测
5. 三元组必须语义完整：(头实体, 关系, 尾实体)
"""

EXTRACTION_FEW_SHOT = """示例输入：
柴胡疏肝散由柴胡、香附、枳壳、陈皮、川芎、白芍、甘草组成，主治肝气郁滞证，症见胁肋疼痛、胸闷善太息。

示例输出：
entities:
- {{"entity_type": "方剂", "entity_name": "柴胡疏肝散"}}
- {{"entity_type": "药材", "entity_name": "柴胡"}}
- {{"entity_type": "药材", "entity_name": "香附"}}
- {{"entity_type": "证型", "entity_name": "肝气郁滞证"}}
- {{"entity_type": "症状", "entity_name": "胁肋疼痛"}}
- {{"entity_type": "症状", "entity_name": "胸闷"}}
triples:
- {{"head": "柴胡疏肝散", "relation": "组成", "tail": "柴胡"}}
- {{"head": "柴胡疏肝散", "relation": "组成", "tail": "香附"}}
- {{"head": "柴胡疏肝散", "relation": "主治", "tail": "肝气郁滞证"}}
- {{"head": "肝气郁滞证", "relation": "对应症状", "tail": "胁肋疼痛"}}
"""
