"""PDF 文本解析（PyPDF2），逐页抽取为章节。"""

from pathlib import Path

from PyPDF2 import PdfReader


def parse_pdf(path: str | Path) -> dict:
    """解析 PDF 文件，返回统一结构化文档。

    若某页无可提取文本（扫描件），content 为空串，
    可交由 data.parser.ocr_parser 处理对应页面图片。
    """
    path = Path(path)
    reader = PdfReader(str(path))

    sections = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append({"section": f"第{i + 1}页", "content": text})

    return {"source": str(path), "format": "pdf", "sections": sections}
