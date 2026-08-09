"""内容生成 Agent：小红书风格图文创作（DeepSeek + 知识图谱溯源 + Pillow 配图）。"""

from pathlib import Path

from common.config import PROJECT_ROOT
from common.llm import get_llm
from retrieval.hybrid_retriever import HybridRetriever

CONTENT_PROMPT = """你是中医药科普内容创作专家，擅长小红书平台风格。请基于【参考资料】创作一篇笔记。

风格要求：
1. 标题吸引人，适当使用 Emoji（✨🌿💡 等）
2. 正文口语化、分段清晰、要点用 emoji 列表呈现
3. 结尾添加 3-5 个热门话题标签（如 #中医药养生 #健康生活）
4. 全文不超过 {max_length} 字
5. 内容必须忠于参考资料，禁止编造功效，文末注明"养生内容仅供参考，具体请遵医嘱"

输出格式：
【标题】
（标题内容）
【正文】
（正文内容）

【参考资料】
{context}

【创作主题】
{topic}"""

_hybrid: HybridRetriever | None = None


def content_generator(state: dict) -> dict:
    """LangGraph 节点：生成小红书风格图文内容。"""
    topic = state["messages"][-1].content

    global _hybrid
    if _hybrid is None:
        _hybrid = HybridRetriever()
    results = _hybrid.search(topic, top_k=5)
    context = _hybrid.build_context(results)

    from common.config import load_config
    max_length = load_config("agent_config.yaml")["content"]["max_length"]

    content = get_llm(temperature=0.7).invoke(
        CONTENT_PROMPT.format(context=context or "（无相关资料，请仅做常识性提醒）",
                              topic=topic, max_length=max_length)
    ).content

    return {"answer": content, "generated_content": content, "topic": topic}


def generate_card(title: str, body: str, out_name: str = "note_card.png") -> Path:
    """用 Pillow 生成文字配图（知识卡片）。"""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1080, 1440
    img = Image.new("RGB", (width, height), "#f7f3ea")
    draw = ImageDraw.Draw(img)

    def load_font(size: int):
        for font_path in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    title_font, body_font = load_font(56), load_font(32)

    draw.text((60, 80), title[:20], font=title_font, fill="#3d2b1f")
    y = 220
    for line in _wrap(body, 28):
        if y > height - 120:
            break
        draw.text((60, y), line, font=body_font, fill="#4a3f35")
        y += 52

    out_dir = PROJECT_ROOT / "web" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    img.save(out_path)
    return out_path


def _wrap(text: str, width: int) -> list[str]:
    lines, buf = [], ""
    for ch in text.replace("\n", "\u3000"):
        buf += ch
        if ch == "\u3000" or len(buf) >= width:
            lines.append(buf)
            buf = ""
    if buf:
        lines.append(buf)
    return lines
