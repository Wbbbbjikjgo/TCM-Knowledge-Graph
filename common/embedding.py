"""达摩院 GTE 中文向量服务：经 ModelScope 魔搭社区下载，国内直连。"""

import numpy as np

from common.config import load_config


class EmbeddingService:
    """单例向量编码器：damo/nlp_gte_sentence-embedding_chinese-base（768维）。"""

    _instance: "EmbeddingService | None" = None

    def __init__(self):
        from modelscope import snapshot_download
        from sentence_transformers import SentenceTransformer

        cfg = load_config("faiss_config.yaml")["embedding"]
        # 首次运行自动从魔搭社区下载模型权重，之后直接读本地缓存
        local_dir = snapshot_download(cfg["model"])
        self.model = SentenceTransformer(local_dir)
        self.dim = cfg["dim"]

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """批量编码并 L2 归一化（归一化后内积 == 余弦相似度）。"""
        if isinstance(texts, str):
            texts = [texts]
        vectors = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")
