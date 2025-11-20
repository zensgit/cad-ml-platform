# CAD ML Platform Docker Image
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgeos-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY knowledge_base/ ./knowledge_base/

# 创建日志目录
RUN mkdir -p /var/log/cad-ml

# 设置环境变量
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO
ENV WORKERS=4

# 暴露端口
EXPOSE 8000
EXPOSE 9090

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

# 运行服务
CMD ["python", "-m", "src.main"]
