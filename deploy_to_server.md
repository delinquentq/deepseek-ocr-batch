# 🚀 DeepSeek OCR 批量处理系统 - 服务器部署指南

## 📋 目录
1. [环境要求](#环境要求)
2. [文件传输](#文件传输)
3. [服务器环境配置](#服务器环境配置)
4. [自动化部署](#自动化部署)
5. [服务化运行](#服务化运行)
6. [监控和维护](#监控和维护)
7. [故障排除](#故障排除)

## 🔧 环境要求

### 服务器硬件要求
- **GPU**: NVIDIA RTX 3090/4090 或 A100 (≥20GB 显存)
- **CPU**: 8核心以上，推荐16核心
- **内存**: 32GB以上，推荐64GB
- **存储**: 200GB以上可用空间，推荐SSD

### 软件环境要求
- **操作系统**: Ubuntu 20.04/22.04 或 CentOS 8+
- **Python**: 3.8+ (推荐3.10)
- **CUDA**: 11.8+ (推荐12.1)
- **NVIDIA驱动**: 470.0+
- **Docker**: 可选，推荐用于隔离环境

## 📤 文件传输

### 方法1：直接SCP传输 (推荐简单部署)

```bash
# 1. 打包项目文件
cd /path/to/DeepSeek-OCR-main/DeepSeek-OCR-master/DeepSeek-OCR-vllm
tar -czf deepseek_ocr_batch.tar.gz \
    *.py *.txt *.md *.sh *.json \
    --exclude="*.pyc" --exclude="__pycache__" \
    --exclude="output_results" --exclude="temp_processing" \
    --exclude="logs"

# 2. 传输到服务器
scp deepseek_ocr_batch.tar.gz user@your-server:/home/user/

# 3. 在服务器上解压
ssh user@your-server
cd /home/user
tar -xzf deepseek_ocr_batch.tar.gz
mv DeepSeek-OCR-vllm deepseek-ocr-batch
cd deepseek-ocr-batch
```

### 方法2：Git同步 (推荐版本控制)

```bash
# 在本地创建Git仓库（如果还没有）
git init
git add .
git commit -m "Initial batch processing system"

# 推送到远程仓库（GitHub/GitLab）
git remote add origin https://github.com/yourusername/deepseek-ocr-batch.git
git push -u origin main

# 在服务器上克隆
ssh user@your-server
git clone https://github.com/yourusername/deepseek-ocr-batch.git
cd deepseek-ocr-batch
```

### 方法3：Rsync同步 (推荐开发模式)

```bash
# 同步文件到服务器
rsync -avz --exclude="*.pyc" --exclude="__pycache__" \
    --exclude="output_results" --exclude="temp_processing" \
    --exclude="logs" --exclude=".git" \
    ./ user@your-server:/home/user/deepseek-ocr-batch/

# 实时同步（开发时使用）
rsync -avz --delete --exclude="*.pyc" --exclude="__pycache__" \
    ./ user@your-server:/home/user/deepseek-ocr-batch/
```

## 🔧 服务器环境配置

### 自动化部署脚本

创建一键部署脚本：