"""中医药知识图谱 Schema 定义。"""

# 实体类型（中文）→ Neo4j 节点标签
ENTITY_LABEL_MAP = {
    "药材": "Herb",
    "方剂": "Fangji",
    "症状": "Symptom",
    "证型": "Syndrome",
    "功效": "Effect",
    "疾病": "Disease",
}

ENTITY_TYPES = list(ENTITY_LABEL_MAP.keys())

# 关系类型（中文）→ Neo4j 关系类型
RELATION_TYPE_MAP = {
    "主治": "TREATS",                    # (方剂)->(证型/疾病)
    "组成": "COMPOSED_OF",               # (方剂)->(药材)
    "功效": "HAS_EFFECT",                # (药材)->(功效)
    "禁忌": "CONTRAINDICATED_WITH",      # (药材)->(药材/证型)
    "归经": "MERIDIAN_TROPISM",          # (药材)->(经络)
    "对应症状": "HAS_SYMPTOM",           # (证型)->(症状)
}

RELATION_TYPES = list(RELATION_TYPE_MAP.keys())

# 原文块节点与“实体出现于原文块”的关系
CHUNK_LABEL = "Chunk"
MENTIONED_IN = "MENTIONED_IN"

# 关系合法性约束：(头实体类型, 关系, 尾实体类型) 允许的组合
VALID_TRIPLE_PATTERNS = {
    ("方剂", "主治", "证型"),
    ("方剂", "主治", "疾病"),
    ("方剂", "组成", "药材"),
    ("药材", "功效", "功效"),
    ("药材", "禁忌", "药材"),
    ("证型", "对应症状", "症状"),
    ("疾病", "对应症状", "症状"),
}
