"""实体与关系抽取器：DeepSeek 结构化输出（实体 + 关系三元组）。"""

from pydantic import BaseModel, Field

from common.llm import get_llm
from kg.entity_extraction.prompt_templates import (
    EXTRACTION_FEW_SHOT,
    EXTRACTION_SYSTEM_PROMPT,
)
from kg.schema import ENTITY_TYPES, RELATION_TYPES


class Entity(BaseModel):
    entity_type: str = Field(description=f"实体类型，只能是{ENTITY_TYPES}之一")
    entity_name: str = Field(description="实体规范名称")


class Triple(BaseModel):
    head: str = Field(description="头实体名称")
    relation: str = Field(description=f"关系类型，只能是{RELATION_TYPES}之一")
    tail: str = Field(description="尾实体名称")


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)


class EntityRelationExtractor:
    """基于 DeepSeek 结构化输出的实体与关系联合抽取器。"""

    def __init__(self):
        self.chain = get_llm(temperature=0).with_structured_output(ExtractionResult)
        self.system_prompt = EXTRACTION_SYSTEM_PROMPT.format(
            entity_types="、".join(ENTITY_TYPES),
            relation_types="、".join(RELATION_TYPES),
        )

    def extract(self, text: str) -> ExtractionResult:
        """从单段文本抽取实体与三元组。"""
        text = text.strip()
        if not text:
            return ExtractionResult()
        messages = [
            ("system", self.system_prompt + "\n" + EXTRACTION_FEW_SHOT),
            ("human", text),
        ]
        return self.chain.invoke(messages)

    def extract_document(self, doc: dict, chunk_size: int = 800) -> ExtractionResult:
        """对 parse_file 输出的结构化文档逐块抽取并合并去重。"""
        merged = ExtractionResult()
        seen_entities, seen_triples = set(), set()

        for section in doc.get("sections", []):
            for chunk in _split_chunks(section.get("content", ""), chunk_size):
                result = self.extract(chunk)
                for e in result.entities:
                    key = (e.entity_type, e.entity_name)
                    if key not in seen_entities:
                        seen_entities.add(key)
                        merged.entities.append(e)
                for t in result.triples:
                    key = (t.head, t.relation, t.tail)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        merged.triples.append(t)
        return merged


def _split_chunks(text: str, chunk_size: int) -> list[str]:
    """按段落边界切分长文本，避免超出模型上下文。"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) > chunk_size and buf:
            chunks.append(buf)
            buf = p
        else:
            buf = buf + "\n" + p if buf else p
    if buf:
        chunks.append(buf)
    return chunks or ([text] if text else [])
