"""中医药演示数据集：10 个经典方剂 + 30+ 药材 + 10 证型 + 20+ 症状 + 15 功效"""

from kg.entity_extraction.extractor import Entity, ExtractionResult, Triple
from kg.neo4j_loader import KnowledgeGraphLoader


# ============ 方剂数据 ============
FORMULAS = [
    {
        "name": "柴胡疏肝散",
        "herbs": ["柴胡", "香附", "枳壳", "陈皮", "川芎", "白芍", "甘草"],
        "syndrome": "肝郁气滞证",
        "symptoms": ["胁肋胀痛", "胸闷", "善太息", "情志抑郁"],
    },
    {
        "name": "四物汤",
        "herbs": ["当归", "熟地黄", "白芍", "川芎"],
        "syndrome": "血虚证",
        "symptoms": ["头晕目眩", "面色萎黄", "心悸失眠", "月经不调"],
    },
    {
        "name": "六味地黄丸",
        "herbs": ["熟地黄", "山茱萸", "山药", "泽泻", "牡丹皮", "茯苓"],
        "syndrome": "肾阴虚证",
        "symptoms": ["腰膝酸软", "头晕耳鸣", "潮热盗汗", "遗精"],
    },
    {
        "name": "麻黄汤",
        "herbs": ["麻黄", "桂枝", "杏仁", "甘草"],
        "syndrome": "风寒表实证",
        "symptoms": ["恶寒发热", "头痛身痛", "无汗", "气喘"],
    },
    {
        "name": "桂枝汤",
        "herbs": ["桂枝", "白芍", "生姜", "大枣", "甘草"],
        "syndrome": "风寒表虚证",
        "symptoms": ["发热", "汗出恶风", "鼻鸣干呕", "头痛"],
    },
    {
        "name": "逍遥散",
        "herbs": ["柴胡", "当归", "白芍", "白术", "茯苓", "甘草", "薄荷", "生姜"],
        "syndrome": "肝郁脾虚证",
        "symptoms": ["胁肋胀痛", "神疲食少", "月经不调", "乳房胀痛"],
    },
    {
        "name": "小青龙汤",
        "herbs": ["麻黄", "桂枝", "干姜", "细辛", "半夏", "五味子", "白芍", "甘草"],
        "syndrome": "外寒内饮证",
        "symptoms": ["恶寒发热", "咳嗽气喘", "痰多清稀", "胸痞"],
    },
    {
        "name": "补中益气汤",
        "herbs": ["黄芪", "党参", "白术", "甘草", "当归", "陈皮", "升麻", "柴胡"],
        "syndrome": "脾胃气虚证",
        "symptoms": ["少气懒言", "体倦肢软", "面色萎黄", "大便溏薄"],
    },
    {
        "name": "天王补心丹",
        "herbs": ["生地黄", "人参", "丹参", "玄参", "茯苓", "五味子", "远志", "桔梗", "当归", "天冬", "麦冬", "柏子仁", "酸枣仁"],
        "syndrome": "心肾不交证",
        "symptoms": ["心烦失眠", "健忘", "梦遗", "手足心热"],
    },
    {
        "name": "血府逐瘀汤",
        "herbs": ["桃仁", "红花", "当归", "生地黄", "川芎", "赤芍", "牛膝", "桔梗", "柴胡", "枳壳", "甘草"],
        "syndrome": "胸中血瘀证",
        "symptoms": ["胸痛", "头痛", "心悸", "失眠多梦"],
    },
]

# ============ 药材功效 ============
HERB_EFFECTS = {
    "柴胡": ["疏肝解郁", "解表退热", "升举阳气"],
    "香附": ["疏肝解郁", "理气宽中", "调经止痛"],
    "枳壳": ["理气宽中", "行滞消胀"],
    "陈皮": ["理气健脾", "燥湿化痰"],
    "川芎": ["活血行气", "祛风止痛"],
    "白芍": ["养血调经", "敛阴止汗", "柔肝止痛"],
    "甘草": ["补脾益气", "清热解毒", "调和诸药"],
    "当归": ["补血活血", "调经止痛", "润肠通便"],
    "熟地黄": ["滋阴补血", "益精填髓"],
    "山茱萸": ["补益肝肾", "收涩固脱"],
    "山药": ["补脾养胃", "生津益肺", "补肾涩精"],
    "泽泻": ["利水渗湿", "泄热"],
    "牡丹皮": ["清热凉血", "活血化瘀"],
    "茯苓": ["利水渗湿", "健脾", "宁心"],
    "麻黄": ["发汗解表", "宣肺平喘", "利水消肿"],
    "桂枝": ["发汗解肌", "温通经脉", "助阳化气"],
    "杏仁": ["降气止咳", "润肠通便"],
    "干姜": ["温中散寒", "回阳通脉", "温肺化饮"],
    "细辛": ["祛风散寒", "通窍止痛", "温肺化饮"],
    "半夏": ["燥湿化痰", "降逆止呕", "消痞散结"],
    "五味子": ["收敛固涩", "益气生津", "补肾宁心"],
    "白术": ["健脾益气", "燥湿利水", "止汗安胎"],
    "薄荷": ["疏散风热", "清利头目", "利咽透疹"],
    "黄芪": ["补气升阳", "固表止汗", "利水消肿", "托毒生肌"],
    "党参": ["补中益气", "健脾益肺"],
    "升麻": ["发表透疹", "清热解毒", "升举阳气"],
    "生地黄": ["清热凉血", "养阴生津"],
    "桃仁": ["活血祛瘀", "润肠通便", "止咳平喘"],
    "红花": ["活血通经", "散瘀止痛"],
    "赤芍": ["清热凉血", "散瘀止痛"],
    "牛膝": ["逐瘀通经", "补肝肾", "强筋骨", "引血下行"],
    "桔梗": ["宣肺", "利咽", "祛痰", "排脓"],
}


def build_extraction_result() -> ExtractionResult:
    """从演示数据构建 ExtractionResult。"""
    entities = []
    triples = []
    seen_entities = set()
    seen_triples = set()

    def add_entity(entity_type: str, entity_name: str):
        key = (entity_type, entity_name)
        if key not in seen_entities:
            seen_entities.add(key)
            entities.append(Entity(entity_type=entity_type, entity_name=entity_name))

    def add_triple(head: str, relation: str, tail: str):
        key = (head, relation, tail)
        if key not in seen_triples:
            seen_triples.add(key)
            triples.append(Triple(head=head, relation=relation, tail=tail))

    # 处理每个方剂
    for formula in FORMULAS:
        fname = formula["name"]
        add_entity("方剂", fname)

        # 方剂 -> 组成 -> 药材
        for herb in formula["herbs"]:
            add_entity("药材", herb)
            add_triple(fname, "组成", herb)

        # 方剂 -> 主治 -> 证型
        syndrome = formula["syndrome"]
        add_entity("证型", syndrome)
        add_triple(fname, "主治", syndrome)

        # 证型 -> 对应症状 -> 症状
        for symptom in formula["symptoms"]:
            add_entity("症状", symptom)
            add_triple(syndrome, "对应症状", symptom)

    # 药材 -> 功效 -> 功效
    for herb, effects in HERB_EFFECTS.items():
        add_entity("药材", herb)
        for effect in effects:
            add_entity("功效", effect)
            add_triple(herb, "功效", effect)

    return ExtractionResult(entities=entities, triples=triples)


def import_demo_data():
    """导入演示数据到 Neo4j。"""
    loader = KnowledgeGraphLoader()
    try:
        loader.ensure_schema()
        result = build_extraction_result()
        stats = loader.load(result)
        print(f"[OK] 演示数据导入完成：{stats['nodes']} 个节点，{stats['relations']} 条关系")
        print(f"   - 方剂：{len(FORMULAS)} 个")
        print(f"   - 药材：{len(HERB_EFFECTS)} 个")
        print(f"   - 证型：{len(set(f['syndrome'] for f in FORMULAS))} 个")
        print(f"   - 症状：{len(set(s for f in FORMULAS for s in f['symptoms']))} 个")
        print(f"   - 功效：{len(set(e for effects in HERB_EFFECTS.values() for e in effects))} 个")
    finally:
        loader.close()


if __name__ == "__main__":
    import_demo_data()
