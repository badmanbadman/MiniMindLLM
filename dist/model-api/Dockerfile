FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖（根据你的模型需求调整）
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和模型
COPY . .

# 创建日志目录
RUN mkdir -p /app/logs

# 暴露端口
EXPOSE 5000

# 健康检查（安装curl）
RUN apt-get update && apt-get install -y curl && apt-get clean

# 启动应用
CMD ["python", "app.py"]