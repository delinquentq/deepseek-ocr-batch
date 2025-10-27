# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DeepSeek OCR 批量处理系统 - PDF文档的OCR识别与结构化数据提取系统。

**处理流程：** PDF → DeepSeek OCR (vLLM) → Markdown+图像 → 规则引擎+LLM (Gemini 2.5 Flash) → JSON Schema v1.3.1

**技术栈：**
- OCR引擎: vLLM 0.8.5 + DeepSeek-OCR (本地GPU)
- 后处理: 规则引擎 + OpenRouter API (Gemini 2.5 Flash)
- 深度学习: PyTorch 2.6.0 + CUDA 11.8
- 目标硬件: RTX 4090 24G (可配置不同硬件)

## 常用命令

### 快速开始

```bash
# 1. 环境配置（首次运行）
conda create -n deepseek-ocr python=3.10 -y
conda activate deepseek-ocr
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_batch.txt

# 2. 设置API密钥（必需）
export OPENROUTER_API_KEY=your_api_key_here

# 3. 运行处理
python run_batch_processor.py              # 处理 input_pdfs/ 下所有PDF
python run_batch_processor.py -y           # 跳过确认
python run_batch_processor.py --input /path/to/pdfs  # 指定输入目录
```

### 批量处理（推荐用法）

```bash
# 后台处理多个目录（生产环境）
python run_batch_processor.py -y \
  --input "/path/to/batch1" \
  > logs/batch1_$(date +%F_%H%M).log 2>&1 &

# 监控处理进度
tail -f logs/batch_processor.log | grep -E "保存JSON|阶段B|失败"

# GPU监控
watch -n 5 nvidia-smi
```

### 测试和调试

```bash
# 系统完整性测试
python test_batch_system.py

# API连接测试
python test_api_connection.py

# 单个模块测试
python test_figure_extraction.py
```

## 核心架构

### 两阶段处理流程

系统采用**异步两阶段流水线**架构，最大化GPU和API利用率：

```
阶段A（生产者）: PDF → OCR → Markdown+图像
  ├─ DeepSeekOCRBatchProcessor (GPU密集)
  ├─ vLLM 推理引擎 (本地GPU)
  ├─ 批量处理: BATCH_SIZE=10, 并发=15
  └─ 输出: output_results/{pdf_name}/{pdf_name}.md + images/
        ↓ (队列传递)
阶段B（消费者）: Markdown → JSON结构化数据
  ├─ MarkdownToJsonEngine (规则引擎，无API调用)
  ├─ BatchFigureProcessor (图表提取，Gemini API)
  ├─ JsonMerger (合并结果)
  └─ 输出: output_report/{pdf_name}/{pdf_name}.json
```

**关键特性：**
- 阶段A和B并行运行，充分利用GPU和网络资源
- 规则引擎处理文本和表格（无成本）
- LLM仅用于图表数据提取（降低80%+ API开销）
- 断点续传：阶段A跳过已有MD，阶段B跳过已有JSON

### 核心文件说明

**主处理器：**
- `batch_pdf_processor.py` - 两阶段协调器（1200+行）
  - `BatchPDFProcessor`: 主控类，管理阶段A和B
  - `DeepSeekOCRBatchProcessor`: OCR批量处理（阶段A）
  - 异步队列管理和错误处理

**后处理引擎（阶段B）：**
- `md_to_json_engine.py` - 规则引擎（提取文本、表格、数值）
- `batch_figure_processor.py` - 图表处理器（Gemini API）
- `json_merger.py` - 结果合并器
- `md_cleaner.py` - Markdown清理器
- `figure_filter.py` - 图表过滤器（跳过股价图等）

**配置和工具：**
- `config_batch.py` - 统一配置管理（硬件、API、路径）
- `deepseek_ocr.py` - vLLM模型集成
- `run_batch_processor.py` - 启动脚本
- `json schema.json` - 输出数据规范 v1.3.1

### 配置调优 (config_batch.py)

**当前配置 (RTX 4090 24G 优化):**
```python
# 硬件配置
GPU_MEMORY_UTILIZATION = 0.85  # 85%显存 (~20GB)
BATCH_SIZE = 10                # 每批10页
MAX_CONCURRENCY = 15           # 最大并发15
NUM_WORKERS = 32               # 32个预处理线程

# 处理配置
MAX_CONCURRENT_PDFS = 6        # 阶段A并发6个PDF
MAX_CONCURRENT_API_CALLS = 15  # 阶段B并发15个API调用（图表提取）
```

**针对不同硬件调整：**

| GPU型号 | GPU_MEM_UTIL | BATCH_SIZE | MAX_CONCURRENCY | 预期速度 |
|---------|--------------|------------|-----------------|----------|
| RTX 3090 24G | 0.75 | 6 | 10 | 基准 |
| RTX 4090 24G | 0.85 | 10 | 15 | 2-3x |
| RTX 4090 48G | 0.90 | 12 | 16 | 3-4x |
| A100 40G | 0.85 | 12 | 16 | 3-4x |

**API限流时降低并发：**
```python
config.processing.MAX_CONCURRENT_API_CALLS = 8  # 从15降到8
```

### 输出目录结构

```
项目根目录/
├── input_pdfs/              # PDF输入
├── output_results/          # 阶段A输出（OCR结果）
│   └── {date}/
│       └── {pdf_name}/
│           ├── {pdf_name}.md         # Markdown文本
│           └── images/               # 提取的图像
│               ├── 0_0.jpg
│               └── 0_1.jpg
├── output_report/           # 阶段B输出（JSON报告）
│   └── {date}/
│       └── {pdf_name}/
│           ├── {pdf_name}.json       # 最终JSON
│           └── {pdf_name}_template.json  # 中间结果
└── temp_processing/         # 临时文件
```

## 开发指南

### 修改阶段B处理逻辑

阶段B由多个引擎组成，各自独立：

1. **修改规则引擎** (md_to_json_engine.py)
   - 调整文本段落提取规则
   - 修改表格解析逻辑
   - 自定义数值提取模式

2. **调整图表处理** (batch_figure_processor.py)
   - 修改图表过滤规则 (figure_filter.py)
   - 调整Gemini提示词
   - 优化API并发策略

3. **自定义合并逻辑** (json_merger.py)
   - 修改字段优先级
   - 添加数据后处理
   - 实现自定义验证

### 调整性能参数

```python
# config_batch.py 中的关键参数

# OCR速度（阶段A）
BATCH_SIZE = 10              # 增大提速但需更多显存
MAX_CONCURRENCY = 15         # GPU并发数

# API速度（阶段B）
MAX_CONCURRENT_API_CALLS = 15  # API并发数（受限于配额）
REQUEST_TIMEOUT = 600        # 单次请求超时
LLM_MAX_TOKENS_IMAGE = 1536  # 图表提取token限制（降低可提速）
```

### 添加新的LLM模型

在 `config_batch.py` 添加：
```python
MODELS = {
    "gemini": "google/gemini-2.5-flash",
    "gpt4": "openai/gpt-4-turbo"  # 新增模型
}
```

在 `batch_figure_processor.py` 中切换模型（约180行）。

## 故障排除

### 常见问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| **CUDA out of memory** | 显存不足 | 降低 `BATCH_SIZE` (10→6) 或 `GPU_MEMORY_UTILIZATION` (0.85→0.75) |
| **API rate limit** | OpenRouter限流 | 降低 `MAX_CONCURRENT_API_CALLS` (15→8) |
| **阶段B卡住不动** | 图表API超时 | 检查网络，查看 `logs/batch_processor.log` |
| **JSON缺少字段** | Schema验证失败 | 检查 `{pdf_name}_template.json` 中间结果 |
| **ImportError: vllm** | 依赖未安装 | `pip install --force-reinstall -r requirements_batch.txt` |
| **阶段A跳过所有PDF** | 已有MD文件 | 删除 `output_results/` 中对应目录 |

### 性能基准

| 文档大小 | RTX 3090 24G | RTX 4090 24G | 备注 |
|---------|--------------|--------------|------|
| 小文档 (10页) | 2-3分钟 | 1-2分钟 | 阶段A+B总时间 |
| 中文档 (30页) | 5-8分钟 | 3-5分钟 | 图表多时B阶段占主导 |
| 大文档 (100页) | 20-30分钟 | 12-18分钟 | 并行处理优势明显 |

**提速关键：** 两阶段并行 + 规则引擎 + 图表过滤

### 日志查看

```bash
# 监控整体进度
tail -f logs/batch_processor.log | grep -E "阶段A|阶段B|完成|失败"

# 查看API错误
grep "API调用失败" logs/batch_processor.log

# 查找特定PDF
grep "report.pdf" logs/batch_processor.log
```

## 环境变量

```bash
# 必需
export OPENROUTER_API_KEY=sk-or-v1-xxxxx

# 可选
export DEEPSEEK_OCR_MODEL_PATH=deepseek-ai/DeepSeek-OCR  # 默认自动下载
export CUDA_VISIBLE_DEVICES=0                             # 指定GPU
export VLLM_USE_V1=0                                      # vLLM版本控制
```

## 重要提醒

1. **首次运行会下载模型**：约10GB，需要HuggingFace访问权限和稳定网络
2. **API配额管理**：当前15并发API调用（仅图表提取），确保OpenRouter配额充足
3. **断点续传**：重新运行会跳过已有MD和JSON，删除输出目录可强制重新处理
4. **监控显存**：使用 `nvidia-smi` 或 `watch -n 1 nvidia-smi` 实时监控
5. **两阶段独立**：阶段A失败不影响阶段B处理其他PDF，反之亦然
6. **日志重要性**：遇到问题先查看 `logs/batch_processor.log`，包含详细错误信息

## 扩展阅读

- `RTX4090_OPTIMIZATION.md` - 硬件优化详细说明
- `README.md` - 项目部署和使用指南
- `json schema.json` - 输出JSON格式规范
