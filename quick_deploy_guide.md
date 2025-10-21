# 🚀 DeepSeek OCR 批量处理系统 - 快速部署指南

## 📋 三步部署到服务器

### 🏠 步骤1: 本地上传到GitHub

```bash
# 在项目目录下
cd "H:\居丽叶玩具项目\DeepSeek-OCR-main\DeepSeek-OCR-master\DeepSeek-OCR-vllm"

# 初始化Git（如果还没有）
git init

# 添加文件
git add .

# 提交
git commit -m "feat: DeepSeek OCR批量处理系统完整版"

# 推送到GitHub（需要先在GitHub创建仓库）
git remote add origin https://github.com/你的用户名/deepseek-ocr-batch.git
git push -u origin main
```

### 🖥️ 步骤2: 服务器一键部署

```bash
# SSH连接到服务器
ssh your-username@your-server-ip

# 下载并运行部署脚本
curl -fsSL https://raw.githubusercontent.com/你的用户名/deepseek-ocr-batch/main/server_deploy.sh | bash

# 或者手动下载运行
wget https://raw.githubusercontent.com/你的用户名/deepseek-ocr-batch/main/server_deploy.sh
chmod +x server_deploy.sh
./server_deploy.sh
```

### ⚙️ 步骤3: 配置和启动

```bash
# 进入项目目录
cd ~/deepseek-ocr-batch

# 配置API密钥
vim .env
# 修改: OPENROUTER_API_KEY=your_actual_api_key

# 放入PDF文件
cp /path/to/your/pdfs/*.pdf input_pdfs/

# 启动处理
./start_service.sh
```

## 🔧 详细部署流程

### 方案A: 自动化部署（推荐）

#### 1. 本地准备
```bash
# 1.1 在GitHub创建新仓库
# 访问 https://github.com/new
# 仓库名: deepseek-ocr-batch
# 设为Public（方便部署脚本下载）

# 1.2 本地推送代码
git remote add origin https://github.com/你的用户名/deepseek-ocr-batch.git
git push -u origin main
```

#### 2. 服务器部署
```bash
# 2.1 连接服务器
ssh -i ~/.ssh/your-key.pem user@your-server-ip

# 2.2 运行一键部署
curl -fsSL https://raw.githubusercontent.com/你的用户名/deepseek-ocr-batch/main/server_deploy.sh | bash
```

### 方案B: 手动部署

#### 1. 克隆项目
```bash
# SSH到服务器
ssh user@your-server

# 克隆项目
git clone https://github.com/你的用户名/deepseek-ocr-batch.git
cd deepseek-ocr-batch
```

#### 2. 环境配置
```bash
# 安装Conda（如果没有）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建环境
conda create -n deepseek-ocr python=3.10 -y
conda activate deepseek-ocr

# 安装依赖
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_batch.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

#### 3. 配置和启动
```bash
# 创建目录
mkdir -p input_pdfs output_results temp_processing logs

# 配置环境变量
cp .env.example .env
vim .env  # 设置OPENROUTER_API_KEY

# 运行测试
python test_batch_system.py

# 启动服务
python run_batch_processor.py
```

## 🔑 关键配置文件

### .env文件配置
```bash
# 必须配置
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxx

# 可选配置
DEEPSEEK_OCR_MODEL_PATH=deepseek-ai/DeepSeek-OCR
CUDA_VISIBLE_DEVICES=0
VLLM_USE_V1=0
```

### config_batch.py调优
```python
# 如果显存不足，调低这些参数
BATCH_SIZE = 2                    # 降低批处理大小
MAX_CONCURRENCY = 4               # 降低并发数
GPU_MEMORY_UTILIZATION = 0.6      # 降低显存使用率
```

## 🚀 快速启动命令

### 基础使用
```bash
# 进入项目目录
cd ~/deepseek-ocr-batch

# 激活环境
conda activate deepseek-ocr

# 环境检查
python run_batch_processor.py --setup

# 开始处理
python run_batch_processor.py

# 后台运行
nohup python run_batch_processor.py > processing.log 2>&1 &

# 查看日志
tail -f logs/batch_processor.log
```

### 高级使用
```bash
# 指定文件处理
python run_batch_processor.py -f report1.pdf -f report2.pdf

# 跳过确认
python run_batch_processor.py -y

# 使用screen/tmux保持会话
screen -S deepseek-ocr
python run_batch_processor.py
# Ctrl+A, D 分离会话
# screen -r deepseek-ocr 重新连接
```

## 📊 监控和维护

### 系统监控
```bash
# GPU使用情况
watch -n 5 nvidia-smi

# 处理进度
tail -f logs/batch_processor.log

# 系统资源
htop

# 存储空间
df -h
```

### 常用维护命令
```bash
# 更新代码
cd ~/deepseek-ocr-batch
git pull origin main

# 重启服务
pkill -f "python.*run_batch_processor.py"
python run_batch_processor.py

# 清理临时文件
rm -rf temp_processing/*

# 备份输出结果
tar -czf results_backup_$(date +%Y%m%d).tar.gz output_results/
```

## 🔧 故障排除

### 常见问题

#### 1. 显存不足
```bash
# 解决方案：降低批处理参数
vim config_batch.py
# 修改: BATCH_SIZE = 2, MAX_CONCURRENCY = 4
```

#### 2. API调用失败
```bash
# 检查API密钥
echo $OPENROUTER_API_KEY

# 测试API连接
python test_batch_system.py
```

#### 3. 依赖安装失败
```bash
# 重新安装
pip install --upgrade pip
pip install --force-reinstall -r requirements_batch.txt
```

#### 4. CUDA版本不匹配
```bash
# 检查CUDA版本
nvcc --version
nvidia-smi

# 重新安装对应版本的PyTorch
pip uninstall torch torchvision
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 性能优化建议

### 服务器配置优化
```bash
# 设置文件句柄限制
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# 设置内存和交换
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf

# 重启生效
sudo reboot
```

### 存储优化
```bash
# 使用SSD存储临时文件
sudo mkdir -p /mnt/ssd/deepseek-tmp
sudo chown $USER:$USER /mnt/ssd/deepseek-tmp
ln -sf /mnt/ssd/deepseek-tmp temp_processing
```

## 📝 部署检查清单

- [ ] GPU驱动和CUDA环境正常
- [ ] Python 3.8+和Conda环境
- [ ] GitHub仓库创建并推送代码
- [ ] 服务器能访问GitHub
- [ ] OPENROUTER_API_KEY已配置
- [ ] 项目依赖包安装完成
- [ ] 系统测试通过
- [ ] 防火墙和网络配置正确
- [ ] 存储空间充足(>200GB)
- [ ] 监控和日志系统正常

完成这个检查清单后，您的DeepSeek OCR批量处理系统就可以在服务器上正常运行了！