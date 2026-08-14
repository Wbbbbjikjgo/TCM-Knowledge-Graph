"""统一配置加载：读取 configs/*.yaml，并支持 .env 环境变量覆盖。"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ModelScope 国产模型缓存目录（达摩院 GTE / COROM）
os.environ.setdefault("MODELSCOPE_CACHE", str(PROJECT_ROOT / "models"))

#lru_cache = Least Recently Used Cache（最近最少使用缓存）  maxsize=None = 不限制缓存大小（无限缓存）
@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    """加载 configs/ 目录下的 YAML 配置文件。"""
    path = PROJECT_ROOT / "configs" / name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) #app_name = config['app_name']  # "MyApp" 直接这样获取


def get_project_root() -> Path:
    return PROJECT_ROOT
