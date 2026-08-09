"""意图识别 Agent：将用户输入路由到对应业务 Agent。"""

from typing import Literal

from pydantic import BaseModel, Field

from common.llm import get_llm

IntentType = Literal["qa", "graph_query", "content_generation", "publish"]

INTENT_PROMPT = """你是意图识别专家。判断用户输入属于以下哪种意图：
- qa：中医药知识问答（如"柴胡有什么功效"、"胸闷胁痛用什么方子"）
- graph_query：查询知识图谱结构化数据（如"列出柴胡疏肝散的组成"、"图谱里有哪些方剂"）
- content_generation：生成科普图文内容（如"帮我写一篇关于艾灸的小红书笔记"）
- publish：发布内容到平台（如"发布这篇笔记"）
只输出意图标签，不要解释。"""


class IntentResult(BaseModel):
    intent: IntentType = Field(description="用户意图类别")


def identify_intent(state: dict) -> dict:
    """LangGraph 节点：解析最后一条用户消息的意图。"""
    messages = state.get("messages", [])
    question = messages[-1].content if messages else ""

    chain = get_llm(temperature=0).with_structured_output(IntentResult)
    try:
        result = chain.invoke([
            ("system", INTENT_PROMPT),
            ("human", question),
        ])
        intent = result.intent
    except Exception:
        intent = "qa"  # 识别失败时默认走问答

    return {"intent": intent}
