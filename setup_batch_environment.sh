#!/bin/bash

# DeepSeek OCR 批量处理系统环境设置脚本
# 针对RTX 3090 24G显存优化

set -e

echo "🚀 DeepSeek OCR 批量处理系统环境设置"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查Python版本
echo -e "${BLUE}🔍 检查Python环境...${NC}"
python_version=$(python --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.8"

if [[ $(echo "$python_version >= $required_version" | bc -l) -eq 1 ]]; then
    echo -e "${GREEN}✓ Python版本: $python_version${NC}"
else
    echo -e "${RED}❌ Python版本过低: $python_version (需要 >= $required_version)${NC}"
    exit 1
fi

# 检查CUDA
echo -e "${BLUE}🔍 检查CUDA环境...${NC}"
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    echo -e "${GREEN}✓ GPU信息: $gpu_info${NC}"

    memory_gb=$(echo $gpu_info | cut -d',' -f2 | xargs)
    if (( memory_gb >= 20000 )); then
        echo -e "${GREEN}✓ 显存充足: ${memory_gb}MB${NC}"
    else
        echo -e "${YELLOW}⚠️  显存可能不足: ${memory_gb}MB (推荐 >= 20GB)${NC}"
    fi
else
    echo -e "${RED}❌ 未检测到NVIDIA GPU或驱动${NC}"
    exit 1
fi

# 创建目录结构
echo -e "${BLUE}📁 创建目录结构...${NC}"
mkdir -p input_pdfs
mkdir -p output_results
mkdir -p temp_processing
mkdir -p logs

echo -e "${GREEN}✓ 目录创建完成${NC}"

# 安装Python依赖
echo -e "${BLUE}📦 安装Python依赖...${NC}"
if [ -f "requirements_batch.txt" ]; then
    pip install -r requirements_batch.txt
    echo -e "${GREEN}✓ 依赖安装完成${NC}"
else
    echo -e "${YELLOW}⚠️  requirements_batch.txt 文件未找到${NC}"
fi

# 检查关键文件
echo -e "${BLUE}📋 检查关键文件...${NC}"

files=(
    "batch_pdf_processor.py"
    "config_batch.py"
    "run_batch_processor.py"
    "json schema.json"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}❌ $file 文件缺失${NC}"
    fi
done

# 环境变量提醒
echo -e "${BLUE}🔑 环境变量配置...${NC}"

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  OPENROUTER_API_KEY 未设置${NC}"
    echo -e "${YELLOW}   请运行: export OPENROUTER_API_KEY=your_api_key${NC}"
else
    echo -e "${GREEN}✓ OPENROUTER_API_KEY 已设置${NC}"
fi

# 创建启动脚本
echo -e "${BLUE}📝 创建启动脚本...${NC}"
cat > start_batch_processor.sh << 'EOF'
#!/bin/bash

# 设置CUDA环境
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0

# 检查API密钥
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ 请设置 OPENROUTER_API_KEY 环境变量"
    echo "   export OPENROUTER_API_KEY=your_api_key"
    exit 1
fi

# 启动批量处理器
python run_batch_processor.py "$@"
EOF

chmod +x start_batch_processor.sh
echo -e "${GREEN}✓ 启动脚本创建完成: start_batch_processor.sh${NC}"

# 创建示例配置文件
echo -e "${BLUE}📄 创建示例配置...${NC}"
cat > .env.example << 'EOF'
# OpenRouter API配置
OPENROUTER_API_KEY=your_openrouter_api_key_here

# DeepSeek OCR模型路径 (可选，默认使用HuggingFace)
DEEPSEEK_OCR_MODEL_PATH=deepseek-ai/DeepSeek-OCR

# CUDA配置
CUDA_VISIBLE_DEVICES=0
VLLM_USE_V1=0

# 可选：Triton配置 (CUDA 11.8)
# TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
EOF

echo -e "${GREEN}✓ 示例配置文件: .env.example${NC}"

# 创建使用说明
cat > README_BATCH.md << 'EOF'
# DeepSeek OCR 批量处理系统使用说明

## 快速开始

### 1. 环境设置
```bash
# 设置API密钥
export OPENROUTER_API_KEY=your_api_key

# 运行环境检查
python run_batch_processor.py --setup
```

### 2. 准备PDF文件
将PDF文件放入 `input_pdfs/` 目录

### 3. 开始处理
```bash
# 方式1：使用启动脚本
./start_batch_processor.sh

# 方式2：直接运行
python run_batch_processor.py

# 方式3：处理指定文件
python run_batch_processor.py -f report1.pdf

# 方式4：跳过确认直接处理
python run_batch_processor.py -y
```

## 输出结果

处理完成后，结果将保存在 `output_results/` 目录：
- `{filename}.md` - Markdown文本
- `{filename}_final.json` - 结构化JSON数据
- `images/` - 提取的图表图像

## 性能优化 (RTX 3090)

系统已针对RTX 3090 24G显存进行优化：
- 批处理大小: 4页/批
- 最大并发数: 6
- 显存利用率: 75%
- 线程池大小: 8

## 故障排除

1. **显存不足**：降低 `BATCH_SIZE` 和 `MAX_CONCURRENCY`
2. **API超时**：增加 `REQUEST_TIMEOUT` 值
3. **JSON验证失败**：检查 `json schema.json` 文件

## 日志查看
```bash
tail -f logs/batch_processor.log
```
EOF

echo -e "${GREEN}✓ 使用说明: README_BATCH.md${NC}"

# 完成提示
echo ""
echo -e "${GREEN}🎉 环境设置完成！${NC}"
echo ""
echo -e "${BLUE}下一步操作:${NC}"
echo -e "1. 设置API密钥: ${YELLOW}export OPENROUTER_API_KEY=your_key${NC}"
echo -e "2. 放入PDF文件到: ${YELLOW}input_pdfs/${NC}"
echo -e "3. 运行处理程序: ${YELLOW}./start_batch_processor.sh${NC}"
echo ""
echo -e "${BLUE}需要帮助? 查看: ${YELLOW}README_BATCH.md${NC}"