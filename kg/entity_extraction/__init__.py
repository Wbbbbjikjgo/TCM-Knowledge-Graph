"""实体抽取模块。"""

from kg.entity_extraction.extractor import (
    Entity,
    EntityRelationExtractor,
    ExtractionResult,
    Triple,
)

__all__ = ["Entity", "Triple", "ExtractionResult", "EntityRelationExtractor"]
