# RTX 4090 快速开始指南

## 🚀 5分钟快速启动

本指南帮助你在 RTX 4090 48G 显卡上快速启动 DeepSeek OCR 批量处理系统。

---

## ✅ 前置检查

### 1. 硬件要求
```bash
# 检查GPU
nvidia-smi

# 应该看到：
# - GPU: NVIDIA GeForce RTX 4090
# - Memory: 48GB
```

### 2. 软件要求
- Python 3.10+
- CUDA 11.8
- Conda (推荐)

---

## 📦 安装步骤

### 步骤 1: 创建环境

```bash
# 创建 conda 环境
conda create -n deepseek-ocr python=3.10 -y
conda activate deepseek-ocr
```

### 步骤 2: 安装 PyTorch

```bash
# 安装 PyTorch (CUDA 11.8)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
```

### 步骤 3: 安装项目依赖

```bash
# 进入项目目录
cd /path/to/deepseek-ocr-batch

# 安装依赖
pip install -r requirements_batch.txt
```

### 步骤 4: 配置环境变量

```bash
# 创建 .env 文件
cat > .env << EOF
OPENROUTER_API_KEY=your_api_key_here
DEEPSEEK_OCR_MODEL_PATH=deepseek-ai/DeepSeek-OCR
CUDA_VISIBLE_DEVICES=0
VLLM_USE_V1=0
EOF

# 或者直接导出
export OPENROUTER_API_KEY=your_api_key_here
```

**获取 OpenRouter API Key:**
1. 访问 https://openrouter.ai/
2. 注册账号
3. 在 Keys 页面创建 API Key

---

## 🧪 测试系统

```bash
# 运行系统测试
python test_batch_system.py
```

**预期输出：**
```
🔬 环境测试...
✅ 环境测试: 通过

🔬 配置验证...
✅ 配置验证: 通过

🔬 JSON Schema验证...
✅ JSON Schema验证: 通过

🔬 GPU显存测试...
✅ GPU显存测试: 通过
  - 可用显存: 48GB
  - 推荐配置: BATCH_SIZE=12, MAX_CONCURRENCY=16
```

---

## 📄 处理第一个PDF

### 1. 准备PDF文件

```bash
# 创建输入目录
mkdir -p input_pdfs

# 复制PDF文件到输入目录
cp /path/to/your/document.pdf input_pdfs/
```

### 2. 运行处理

```bash
# 处理所有PDF
python run_batch_processor.py

# 或者处理指定文件
python run_batch_processor.py -f document.pdf

# 跳过确认提示
python run_batch_processor.py -y
```

### 3. 查看结果

```bash
# OCR结果（Markdown + 图像）
ls -lh output_results/document/

# JSON报告
ls -lh output_report/document/

# 查看JSON内容
cat output_report/document/document.json | jq .
```

---

## 📊 预期性能

### RTX 4090 48G 性能指标

| 文档大小 | 处理时间 | 显存使用 |
|---------|---------|---------|
| 小文档 (5-10页) | 40-60秒 | 38-40GB |
| 中文档 (20-30页) | 90-150秒 | 40-42GB |
| 大文档 (50+页) | 4-6分钟 | 42-43GB |

### 实时监控

```bash
# 终端1: 监控GPU
watch -n 1 nvidia-smi

# 终端2: 监控日志
tail -f logs/batch_processor.log

# 终端3: 监控系统资源
htop
```

---

## 🎯 优化配置

### 当前配置（极速模式）

文件：`config_batch.py`

```python
# RTX 4090 48G 极速配置
GPU_MEMORY_UTILIZATION = 0.90  # 90% 显存利用率
MAX_CONCURRENCY = 16           # 16 并发请求
BATCH_SIZE = 12                # 每批12页
NUM_WORKERS = 24               # 24 预处理线程
MAX_CONCURRENT_PDFS = 6        # 6 并发PDF
MAX_CONCURRENT_API_CALLS = 12  # 12 并发API
```

### 如果遇到问题

**显存不足 (OOM):**
```python
# 降低配置
config.hardware.BATCH_SIZE = 8
config.hardware.MAX_CONCURRENCY = 12
config.hardware.GPU_MEMORY_UTILIZATION = 0.85
```

**API限流:**
```python
# 降低API并发
config.processing.MAX_CONCURRENT_API_CALLS = 8
config.api.RETRY_DELAY_BASE = 2
```

---

## 📁 输出目录结构

```
项目根目录/
├── input_pdfs/              # 输入PDF
│   └── document.pdf
│
├── output_results/          # OCR结果（MD + 图像）
│   └── document/
│       ├── document.md
│       └── images/
│           ├── 0_0.jpg
│           └── 0_1.jpg
│
├── output_report/           # JSON报告（新增）
│   └── document/
│       ├── document.json            # Schema格式JSON
│       └── document_template.json   # 模板格式JSON
│
└── logs/                    # 日志
    └── batch_processor.log
```

---

## 🔧 常见问题

### Q1: 首次运行很慢？
**A:** 首次运行会从 HuggingFace 下载模型（约10GB），需要稳定网络。后续运行会使用缓存。

### Q2: 如何加速处理？
**A:**
1. 确保使用 SSD 存储
2. 提高网络带宽（API调用）
3. 使用激进配置（见优化配置）

### Q3: JSON验证失败？
**A:**
1. 检查 `json schema.json` 文件存在
2. 查看 `logs/batch_processor.log` 详细错误
3. 启用自动修复：`config.validation.AUTO_FIX_SCHEMA_ERRORS = True`

### Q4: 如何批量处理多个PDF？
**A:**
```bash
# 将所有PDF放入 input_pdfs/
cp /path/to/pdfs/*.pdf input_pdfs/

# 运行批量处理
python run_batch_processor.py -y

# 后台运行
nohup python run_batch_processor.py -y > processing.log 2>&1 &
```

### Q5: 如何查看处理进度？
**A:**
```bash
# 实时日志
tail -f logs/batch_processor.log

# 统计已处理文件
ls output_report/ | wc -l

# 查看GPU使用
nvidia-smi
```

---

## 🎓 进阶使用

### 1. 后台运行

```bash
# 使用 nohup
nohup python run_batch_processor.py > processing.log 2>&1 &

# 使用 screen
screen -S deepseek-ocr
python run_batch_processor.py
# Ctrl+A, D 分离会话
# screen -r deepseek-ocr  # 重新连接

# 使用 tmux
tmux new -s deepseek-ocr
python run_batch_processor.py
# Ctrl+B, D 分离会话
# tmux attach -t deepseek-ocr  # 重新连接
```

### 2. 定时任务

```bash
# 添加到 crontab
crontab -e

# 每天凌晨2点处理
0 2 * * * cd /path/to/deepseek-ocr-batch && /path/to/conda/envs/deepseek-ocr/bin/python run_batch_processor.py -y >> /path/to/cron.log 2>&1
```

### 3. 监控脚本

```bash
# 创建监控脚本
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
    echo ""
    echo "=== Processing Status ==="
    tail -n 5 logs/batch_processor.log
    echo ""
    echo "=== Output Count ==="
    echo "OCR Results: $(ls output_results/ 2>/dev/null | wc -l)"
    echo "JSON Reports: $(ls output_report/ 2>/dev/null | wc -l)"
    sleep 5
done
EOF

chmod +x monitor.sh
./monitor.sh
```

---

## 📚 更多资源

- **详细文档:** `CLAUDE.md`
- **优化说明:** `RTX4090_OPTIMIZATION.md`
- **项目说明:** `README.md`
- **配置文件:** `config_batch.py`
- **JSON Schema:** `json schema.json`

---

## 🆘 获取帮助

遇到问题时，请提供以下信息：

```bash
# 1. 系统信息
python test_batch_system.py > system_info.txt

# 2. 错误日志
tail -n 100 logs/batch_processor.log > error_log.txt

# 3. GPU状态
nvidia-smi > gpu_status.txt

# 4. 环境信息
conda list > conda_env.txt
pip list > pip_env.txt
```

---

## ✨ 快速命令参考

```bash
# 环境激活
conda activate deepseek-ocr

# 测试系统
python test_batch_system.py

# 处理PDF
python run_batch_processor.py

# 后台运行
nohup python run_batch_processor.py -y > processing.log 2>&1 &

# 监控GPU
watch -n 1 nvidia-smi

# 查看日志
tail -f logs/batch_processor.log

# 查看结果
ls -lh output_report/

# 验证JSON
python -m json.tool output_report/test/test.json

# 停止处理
pkill -f "python.*run_batch_processor.py"
```

---

**🎉 恭喜！你已经成功启动 DeepSeek OCR 批量处理系统！**

**性能提示：** RTX 4090 48G 配置下，预计处理速度比 RTX 3090 24G 快 **3-4倍**！

**下一步：** 阅读 `RTX4090_OPTIMIZATION.md` 了解更多优化技巧。
