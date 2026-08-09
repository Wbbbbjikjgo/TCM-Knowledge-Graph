"""关系抽取后处理：基于图谱 Schema 的规则校验与过滤。"""

from kg.entity_extraction.extractor import Entity, ExtractionResult
from kg.schema import ENTITY_TYPES, RELATION_TYPES, VALID_TRIPLE_PATTERNS


def validate_extraction(result: ExtractionResult) -> ExtractionResult:
    """过滤非法实体类型/关系类型，按模式约束校验三元组，
    并对缺失类型的头尾实体尝试依据关系模式推断类型。"""
    entity_type_of = {}
    valid_entities = []
    for e in result.entities:
        if e.entity_type in ENTITY_TYPES and e.entity_name.strip():
            valid_entities.append(e)
            entity_type_of[e.entity_name] = e.entity_type

    def infer_type(name: str, patterns: list[tuple], pos: int) -> str | None:
        """在模式唯一时推断实体类型，并补充实体记录。"""
        candidates = {p[pos] for p in patterns}
        if len(candidates) == 1:
            etype = candidates.pop()
            entity_type_of[name] = etype
            valid_entities.append(Entity(entity_type=etype, entity_name=name))
            return etype
        return None

    valid_triples = []
    for t in result.triples:
        if t.relation not in RELATION_TYPES or t.head == t.tail:
            continue
        patterns = [p for p in VALID_TRIPLE_PATTERNS if p[1] == t.relation]
        head_type = entity_type_of.get(t.head) or infer_type(t.head, patterns, 0)
        tail_type = entity_type_of.get(t.tail) or infer_type(t.tail, patterns, 2)
        # 头尾实体必须可确定类型，且三元组模式符合 Schema 约束
        if head_type and tail_type and (head_type, t.relation, tail_type) in patterns:
            valid_triples.append(t)

    return ExtractionResult(entities=valid_entities, triples=valid_triples)
