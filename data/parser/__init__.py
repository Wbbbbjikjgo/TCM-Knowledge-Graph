"""文档解析模块：PDF / 扫描件 OCR / 纯文本，输出统一结构化 JSON。"""

import json
from pathlib import Path

from common.config import PROJECT_ROOT

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
PARSED_DIR = RAW_DATA_DIR / "parsed"


def save_parsed(doc: dict, out_dir: Path = PARSED_DIR) -> Path:
    """将解析结果保存为结构化 JSON，返回文件路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(doc["source"]).stem + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return out_path


def parse_file(path: str | Path) -> dict:
    """按扩展名分发解析器，输出统一格式：
    {"source": ..., "format": ..., "sections": [{"section": ..., "content": ...}]}
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from data.parser.pdf_parser import parse_pdf
        return parse_pdf(path)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
        from data.parser.ocr_parser import parse_image
        return parse_image(path)
    if suffix in {".txt", ".md"}:
        return {
            "source": str(path),
            "format": suffix.lstrip("."),
            "sections": [{"section": "全文",
                          "content": path.read_text(encoding="utf-8", errors="ignore")}],
        }
    raise ValueError(f"不支持的文件格式: {suffix}")
