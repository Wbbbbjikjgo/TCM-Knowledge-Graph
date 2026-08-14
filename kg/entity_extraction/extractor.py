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
        self.chain = get_llm(temperature=0).with_structured_output(
            ExtractionResult, method="function_calling"
        )
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
        """
            doc = {
            "sections": [
            {
            "content": "Albert Einstein was born in Germany. He developed the theory of relativity."
            },
            {
            "content": "Einstein won the Nobel Prize in Physics in 1921. His famous equation is E=mc^2."
            }
            ]
            }
        """
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
    """按段落边界切分长文本，避免超出模型上下文。
    
    功能说明：
    1. 将文本按段落（换行符）分割
    2. 尽可能保持段落完整，避免在段落中间截断
    3. 当累积内容超过 chunk_size 时，切分为一个块
    4. 保证每个块的大小尽可能接近但不超过 chunk_size
    
    参数:
        text: 待切分的原始文本
        chunk_size: 每个块的最大字符数
        
    返回:
        list[str]: 切分后的文本块列表
        
    示例：
        输入: "第一段内容。\n第二段内容。\n第三段内容。"
        chunk_size: 10
        输出: ["第一段内容。", "第二段内容。", "第三段内容。"]
    """
    
    # ==================== 第一步：分割段落 ====================
    # 按换行符分割文本，去除空行和首尾空白
    # 例子：
    # text = "麻黄，味辛，微苦，性温。\n\n归肺经、膀胱经。\n具有发汗解表，宣肺平喘的功效。"
    # paragraphs = ["麻黄，味辛，微苦，性温。", "归肺经、膀胱经。", "具有发汗解表，宣肺平喘的功效。"]
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    # ==================== 第二步：初始化变量 ====================
    # chunks: 存储所有切分好的文本块
    # buf: 缓冲区，用于累积当前正在构建的文本块
    chunks, buf = [], ""
    
    # ==================== 第三步：遍历段落构建块 ====================
    for p in paragraphs:
        # 检查当前缓冲区和当前段落组合后是否超过 chunk_size
        # 且缓冲区不为空（避免空块）
        # 
        # 例子1：buf="麻黄，味辛，微苦，性温。" (12字符), p="归肺经、膀胱经。" (8字符), chunk_size=20
        # len(buf) + len(p) = 20, 不超过20，执行 else 分支：合并
        # 
        # 例子2：buf="麻黄，味辛，微苦，性温。归肺经、膀胱经。" (20字符), p="具有发汗解表，宣肺平喘的功效。" (13字符), chunk_size=20
        # len(buf) + len(p) = 33 > 20，执行 if 分支：保存当前块，重置缓冲区
        if len(buf) + len(p) > chunk_size and buf:
            # 将当前缓冲区内容作为一个块保存
            chunks.append(buf)
            # 当前段落成为新块的开始
            buf = p
        else:
            # 合并当前段落到缓冲区
            # 如果缓冲区为空，直接赋值；否则用换行符连接
            buf = buf + "\n" + p if buf else p
    
    # ==================== 第四步：处理最后一个块 ====================
    # 如果缓冲区还有内容，添加到结果中
    if buf:
        chunks.append(buf)
    
    # ==================== 第五步：保底处理 ====================
    # 如果最终没有得到任何块（比如输入为空或只有空行），
    # 返回包含原始文本的单元素列表（如果文本非空），否则返回空列表
    return chunks or ([text] if text else [])
