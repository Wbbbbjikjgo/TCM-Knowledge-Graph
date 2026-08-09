"""DeepSeek LLM 工厂：API 形式接入，国产模型零本地下载。"""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.3, model: str | None = None) -> ChatOpenAI:
    """创建 DeepSeek Chat 客户端（兼容 OpenAI 协议）。"""
    return ChatOpenAI(
        model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=temperature,
    )
