# DeepSeek OCR 批量处理系统 - 完整设置指南

## 🎯 项目概述

基于您的需求，我已经开发了一个完整的批量PDF处理系统，具备以下特性：

- ✅ **批量PDF处理** - 支持多文件并行处理
- ✅ **RTX 3090 24G优化** - 针对您的显卡配置进行内存和并发优化
- ✅ **双模型对比** - Gemini 2.5 Flash + Qwen3-VL-30B同时处理
- ✅ **严格JSON验证** - 基于您的schema.json进行数据验证
- ✅ **图表无损提取** - 确保数据可重建为可视化图表
- ✅ **数据库兼容** - 输出格式完全符合数据库导入要求

## 🚀 快速开始

### 步骤1: 环境准备

```bash
# 1. 安装依赖
pip install -r requirements_batch.txt

# 2. 设置环境变量
export OPENROUTER_API_KEY=your_openrouter_api_key

# 3. 运行环境设置脚本（Linux/Mac）
chmod +x setup_batch_environment.sh
./setup_batch_environment.sh

# 4. 创建必要目录
mkdir -p input_pdfs output_results temp_processing logs
```

### 步骤2: 配置验证

```bash
# 运行系统测试
python test_batch_system.py

# 检查环境配置
python run_batch_processor.py --setup
```

### 步骤3: 开始处理

```bash
# 1. 将PDF文件放入 input_pdfs/ 目录

# 2. 启动批量处理
python run_batch_processor.py

# 或使用启动脚本
./start_batch_processor.sh
```

## 📁 文件结构

```
DeepSeek-OCR-vllm/
├── batch_pdf_processor.py      # 主处理器（核心）
├── config_batch.py             # 配置文件
├── run_batch_processor.py      # 启动脚本
├── test_batch_system.py        # 系统测试
├── requirements_batch.txt      # 依赖包
├── setup_batch_environment.sh  # 环境设置
├── json schema.json            # JSON Schema验证
├── json结构范式说明.md          # 数据格式说明
├── input_pdfs/                 # PDF输入目录
├── output_results/             # 处理结果输出
├── temp_processing/            # 临时文件
└── logs/                       # 日志文件
```

## ⚙️ 系统架构

### 处理流程

```
PDF文件 → DeepSeek OCR → Markdown + 图像 → OpenRouter双模型 → JSON验证 → 数据库格式
```

### 关键组件

1. **DeepSeekOCRBatchProcessor** - PDF OCR处理
2. **OpenRouterProcessor** - 双模型API调用
3. **JSONSchemaValidator** - 数据验证和修复
4. **BatchPDFProcessor** - 主处理协调器

### RTX 3090 优化配置

```python
# 针对24G显存的优化参数
BATCH_SIZE = 4                    # 每批4页
MAX_CONCURRENCY = 6               # 最大并发6个
GPU_MEMORY_UTILIZATION = 0.75     # 显存利用率75%
NUM_WORKERS = 8                   # 8个处理线程
```

## 🔧 配置说明

### 硬件配置 (config_batch.py)

```python
class HardwareConfig:
    GPU_MEMORY_UTILIZATION = 0.75  # 保留25%显存给后续LLM
    MAX_CONCURRENCY = 6             # 并发数
    BATCH_SIZE = 4                  # 批处理大小
    NUM_WORKERS = 8                 # 预处理线程数
```

### API配置

```python
class APIConfig:
    MODELS = {
        "gemini": "google/gemini-2.5-flash",
        "qwen": "qwen/qwen-2.5-vl-72b-instruct"
    }
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 300
```

## 📊 使用示例

### 基本使用

```bash
# 处理所有PDF文件
python run_batch_processor.py

# 处理指定文件
python run_batch_processor.py -f report1.pdf -f report2.pdf

# 跳过确认直接处理
python run_batch_processor.py -y

# 仅检查环境
python run_batch_processor.py --setup
```

### 高级配置

```python
# 修改批处理大小（显存不足时）
config.hardware.BATCH_SIZE = 2
config.hardware.MAX_CONCURRENCY = 4

# 修改API超时时间
config.api.REQUEST_TIMEOUT = 600

# 启用质量检查
config.processing.ENABLE_QUALITY_CHECK = True
```

## 📈 输出格式

### 目录结构

```
output_results/
└── filename/
    ├── filename.md              # Markdown文本
    ├── filename_final.json      # 最终JSON数据
    ├── images/                  # 提取的图表
    │   ├── 0_0.jpg
    │   ├── 0_1.jpg
    │   └── ...
    └── processing_log.txt       # 处理日志
```

### JSON输出格式

严格按照您的`json schema.json`格式：

```json
{
  "_id": "unique_hash_id",
  "source": {
    "file_name": "document.pdf",
    "processing_metadata": {
      "vision_model": "deepseek-ai/DeepSeek-OCR",
      "synthesis_model": "google/gemini-2.5-flash",
      "validation_model": "qwen/qwen-2.5-vl-72b-instruct",
      "processed_at": "2024-10-21T15:30:45Z",
      "pages_processed": 25,
      "successful_pages": 24
    }
  },
  "report": {
    "title": "Financial Analysis Report",
    "report_date": "2024-10-15",
    "report_type": "company",
    "symbols": ["AAPL"],
    "sector": "Technology",
    "content": "Complete synthesized content...",
    "word_count": 3542
  },
  "data": {
    "figures": [
      {
        "figure_id": "revenue_growth_chart",
        "type": "bar_chart",
        "title": "Quarterly Revenue Growth",
        "description": "YoY growth by quarter",
        "data": {
          "labels": ["Q1", "Q2", "Q3", "Q4"],
          "series": [{
            "name": "Growth Rate",
            "values": [12.5, 15.3, 14.8, 18.2],
            "unit": "%"
          }]
        },
        "source_page": 8
      }
    ],
    "numerical_data": [...],
    "companies": [...],
    "key_metrics": [...],
    "extraction_summary": {...}
  },
  "query_capabilities": {...}
}
```

## 🔍 质量保证

### 数据验证

1. **Schema验证** - 严格按照JSON Schema验证
2. **数据完整性** - 确保所有图表包含完整data字段
3. **关联性检查** - 验证figure_id关联关系
4. **类型检查** - 确保数据类型正确

### 双模型对比

- Gemini 2.5 Flash (优先选择)
- Qwen3-VL-30B (备用选择)
- 自动选择质量更好的结果

### 错误处理

- 自动重试机制（最多3次）
- 优雅降级处理
- 详细错误日志

## 📋 常见问题

### Q1: 显存不足怎么办？

```python
# 降低批处理参数
config.hardware.BATCH_SIZE = 2
config.hardware.MAX_CONCURRENCY = 4
config.hardware.GPU_MEMORY_UTILIZATION = 0.6
```

### Q2: API调用失败？

```bash
# 检查API密钥
echo $OPENROUTER_API_KEY

# 测试API连接
python test_batch_system.py
```

### Q3: JSON验证失败？

- 检查`json schema.json`文件完整性
- 启用自动修复: `config.validation.AUTO_FIX_SCHEMA_ERRORS = True`

### Q4: 处理速度慢？

- 增加并发数（显存允许的情况下）
- 使用更快的模型
- 减少PDF DPI设置

## 🔧 性能调优

### RTX 3090优化建议

```python
# 最佳性能配置
BATCH_SIZE = 4                    # 平衡显存和速度
MAX_CONCURRENCY = 6               # 充分利用显存
GPU_MEMORY_UTILIZATION = 0.75     # 为API调用保留显存
NUM_WORKERS = 8                   # 与CPU核心数匹配
```

### 成本优化

```python
# 降低API成本
LLM_MAX_TOKENS = 4000            # 减少token使用
ENABLE_QUALITY_CHECK = False     # 跳过质量检查（不推荐）
```

## 📊 监控和日志

### 日志查看

```bash
# 实时查看处理日志
tail -f logs/batch_processor.log

# 查看错误日志
tail -f logs/errors.log

# 查看测试报告
cat test_report.json
```

### 性能监控

- 处理时间统计
- 显存使用监控
- API调用成功率
- 数据质量指标

## 🎯 下一步计划

1. **数据库集成** - 添加直接数据库导入功能
2. **Web界面** - 开发可视化管理界面
3. **更多模型支持** - 集成更多OCR和LLM模型
4. **实时处理** - 支持实时PDF流处理

## 📞 技术支持

如遇问题请提供：
1. 错误日志 (`logs/batch_processor.log`)
2. 系统配置 (`python test_batch_system.py`)
3. 环境信息 (`python run_batch_processor.py --setup`)

## 📄 许可证

本项目基于原DeepSeek-OCR项目开发，遵循相同的开源许可证。