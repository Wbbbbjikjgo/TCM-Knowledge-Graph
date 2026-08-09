# 中医药知识图谱智能问答系统 - 应用镜像
FROM python:3.10-slim

# Tesseract OCR（含简体中文语言包，用于古籍扫描件解析）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖利用 Docker 层缓存（使用清华镜像源加速国内安装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "web/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
