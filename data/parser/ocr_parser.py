"""扫描件/图片 OCR 解析（Tesseract，需安装 chi_sim 语言包）。"""

from pathlib import Path

import pytesseract
from PIL import Image


def ocr_image(path: str | Path, lang: str = "chi_sim+eng") -> str:
    """对单张图片执行 OCR，返回识别文本。"""
    with Image.open(path) as img:
        # 灰度化可提升古籍扫描件识别率
        return pytesseract.image_to_string(img.convert("L"), lang=lang).strip()


def parse_image(path: str | Path) -> dict:
    """解析图片文件为统一结构化文档。"""
    path = Path(path)
    return {
        "source": str(path),
        "format": "ocr",
        "sections": [{"section": "OCR识别", "content": ocr_image(path)}],
    }
