# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DeepSeek OCR 批量处理系统 - 基于 DeepSeek-OCR + OpenRouter 的智能文档批量处理系统，**针对 RTX 4090 48G 显存极速优化**。

**核心功能：** PDF文档 → OCR识别 → Markdown提取 → 结构化JSON输出（符合金融报告提取Schema v1.3.1）

**技术栈：**
- **OCR引擎：** vLLM 0.8.5 + DeepSeek-OCR 模型
- **后处理：** OpenRouter API (Gemini 2.5 Flash)
- **验证：** JSON Schema v1.3.1 严格验证
- **深度学习：** PyTorch 2.6.0 + CUDA 11.8

**性能：** 相比 RTX 3090 配置提升 **3-4倍** 处理速度

## 常用命令

### 环境设置

```bash
# 创建conda环境
conda create -n deepseek-ocr python=3.10 -y
conda activate deepseek-ocr

# 安装PyTorch (CUDA 11.8)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements_batch.txt

# 配置环境变量（必需）
export OPENROUTER_API_KEY=your_api_key_here
export DEEPSEEK_OCR_MODEL_PATH=deepseek-ai/DeepSeek-OCR
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0
```

### 系统测试

```bash
# 运行完整系统测试（推荐首次运行）
python test_batch_system.py

# 检查环境配置
python run_batch_processor.py --setup
```

### 批量处理

```bash
# 处理 input_pdfs/ 目录下的所有PDF
python run_batch_processor.py

# 处理指定文件
python run_batch_processor.py -f report1.pdf -f report2.pdf

# 跳过确认提示
python run_batch_processor.py -y

# 后台运行
nohup python run_batch_processor.py > processing.log 2>&1 &

# 使用screen保持会话
screen -S deepseek-ocr
python run_batch_processor.py
# Ctrl+A, D 分离会话
```

### 监控和调试

```bash
# 实时查看处理日志
tail -f logs/batch_processor.log

# 监控GPU状态
watch -n 5 nvidia-smi

# 查看系统资源
htop

# 检查存储空间
df -h
```

### 维护命令

```bash
# 清理临时文件
rm -rf temp_processing/*

# 清理日志
rm -rf logs/*.log

# 备份处理结果
tar -czf backup_$(date +%Y%m%d).tar.gz output_results/
```

## 核心架构

### 处理流程

```
PDF输入 (input_pdfs/)
    ↓
DeepSeekOCRBatchProcessor (deepseek_ocr.py + batch_pdf_processor.py)
    ├─ vLLM推理引擎加载模型
    ├─ PDF页面批量处理 (BATCH_SIZE=4)
    ├─ 动态图像裁剪 (CROP_MODE=True)
    └─ Markdown + 图像提取
    ↓
OpenRouterProcessor (batch_pdf_processor.py:400+)
    ├─ AsyncOpenAI客户端
    ├─ Gemini 2.5 Flash API调用
    ├─ 结构化数据提取
    └─ 指数退避重试机制
    ↓
JSONSchemaValidator (batch_pdf_processor.py:500+)
    ├─ Schema v1.3.1 严格验证
    ├─ 自动修复常见错误
    └─ 数据完整性检查
    ↓
输出 (output_results/)
    ├─ {filename}.md - Markdown文本
    ├─ {filename}_final.json - 结构化JSON
    ├─ images/ - 提取的图表
    └─ processing_log.txt - 处理日志
```

### 关键模块

**1. 配置管理 (config_batch.py)**
- `HardwareConfig`: GPU显存优化参数 (RTX 3090 24G)
- `OCRConfig`: DeepSeek OCR模型配置
- `APIConfig`: OpenRouter API配置
- `PathConfig`: 文件路径管理
- `ProcessingConfig`: 处理流程控制
- `ValidationConfig`: 数据验证规则

**2. OCR处理器 (deepseek_ocr.py)**
- `DeepseekOCRForCausalLM`: vLLM模型实现
- `DeepseekOCRProcessor`: 图像预处理器 (process/image_process.py)
- `NoRepeatNGramLogitsProcessor`: N-gram重复检测 (process/ngram_norepeat.py)
- 视觉编码器: SAM-ViT-B + CLIP-L (deepencoder/)

**3. 批量处理器 (batch_pdf_processor.py)**
- `BatchPDFProcessor`: 主处理器协调类
- `DeepSeekOCRBatchProcessor`: OCR批量处理
- `OpenRouterProcessor`: API调用和数据提取
- `JSONSchemaValidator`: Schema验证器

**4. 启动脚本 (run_batch_processor.py)**
- 命令行参数解析
- 环境检查和验证
- 用户交互界面
- 批量任务调度

### 配置参数调优

**RTX 4090 48G 极速配置 (config_batch.py:14-30):**
```python
GPU_MEMORY_UTILIZATION = 0.90  # 充分利用90%显存（43.2GB）
MAX_CONCURRENCY = 16           # 大幅提升并发数
BATCH_SIZE = 12                # 每批处理12页（3倍提升）
NUM_WORKERS = 24               # 预处理线程数（3倍提升）
MAX_CONCURRENT_PDFS = 6        # 并发处理6个PDF
MAX_CONCURRENT_API_CALLS = 12  # 并发12个API调用
```

**显存不足时调整 (降低显存使用):**
```python
config.hardware.BATCH_SIZE = 8
config.hardware.MAX_CONCURRENCY = 12
config.hardware.GPU_MEMORY_UTILIZATION = 0.85
config.processing.MAX_CONCURRENT_PDFS = 4
```

**追求极致速度 (激进配置，风险较高):**
```python
config.hardware.GPU_MEMORY_UTILIZATION = 0.95
config.hardware.MAX_CONCURRENCY = 20
config.hardware.BATCH_SIZE = 16
config.processing.MAX_CONCURRENT_PDFS = 8
config.processing.MAX_CONCURRENT_API_CALLS = 16
```

**API限流时调整:**
```python
config.processing.MAX_CONCURRENT_API_CALLS = 8
config.api.RETRY_DELAY_BASE = 2
config.api.REQUEST_TIMEOUT = 900  # 15分钟
```

### JSON Schema 结构

输出JSON严格遵循 `json schema.json` (v1.3.1)，主要字段：

```json
{
  "schema_version": "1.3.1",
  "doc": {
    "doc_id": "文档hash ID",
    "title": "文档标题",
    "timestamps": { "ingested_at": "...", "extracted_at": "..." },
    "extraction_run": {
      "vision_model": "deepseek-ai/DeepSeek-OCR",
      "synthesis_model": "google/gemini-2.5-flash",
      "pipeline_steps": ["ocr", "extraction", "validation"]
    }
  },
  "passages": [...],  // 文本段落
  "entities": {...},  // 实体识别
  "data": {
    "figures": [...],      // 图表数据（可重建）
    "tables": [...],       // 表格数据
    "numerical_data": [...], // 数值数据
    "companies": [...],    // 公司信息
    "key_metrics": [...]   // 关键指标
  }
}
```

## 开发指南

### 添加新的LLM模型

在 `config_batch.py:66-68` 添加模型配置：
```python
MODELS = {
    "gemini": "google/gemini-2.5-flash",
    "new_model": "provider/model-name"  # 新增
}
```

在 `batch_pdf_processor.py` 的 `OpenRouterProcessor.__init__()` 中使用：
```python
self.model = config.api.MODELS["new_model"]
```

### 自定义验证规则

扩展 `JSONSchemaValidator` 类 (batch_pdf_processor.py:500+)：
```python
def custom_validation(self, data: Dict) -> Tuple[bool, List[str]]:
    """自定义验证逻辑"""
    errors = []
    # 添加自定义检查
    if not data.get("data", {}).get("figures"):
        errors.append("缺少图表数据")
    return len(errors) == 0, errors
```

### 修改OCR提示词

编辑 `config.py:PROMPT` 或 `config_batch.py:37`：
```python
PROMPT = '<image>\n<|grounding|>Your custom prompt here.'
```

### 调试技巧

**1. 启用详细日志 (config_batch.py:133):**
```python
LOG_LEVEL = "DEBUG"
```

**2. 保存中间结果 (config_batch.py:118):**
```python
SAVE_INTERMEDIATE_RESULTS = True
SAVE_RAW_RESPONSES = True
```

**3. 单文件测试:**
```python
# 在 batch_pdf_processor.py 中直接运行
if __name__ == "__main__":
    processor = BatchPDFProcessor()
    asyncio.run(processor.process_single_pdf("test.pdf"))
```

**4. 跳过Schema验证（仅调试）:**
```python
config.validation.STRICT_SCHEMA_VALIDATION = False
```

## 故障排除

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `CUDA out of memory` | 显存不足 | 降低 `BATCH_SIZE` 和 `MAX_CONCURRENCY` |
| `API rate limit exceeded` | OpenRouter限流 | 增加 `RETRY_DELAY_BASE`，降低 `MAX_CONCURRENT_API_CALLS` |
| `JSON validation failed` | Schema不匹配 | 检查 `json schema.json`，启用 `AUTO_FIX_SCHEMA_ERRORS` |
| `Model loading timeout` | 模型下载慢 | 预先下载模型到本地，设置 `DEEPSEEK_OCR_MODEL_PATH` |
| `ImportError: vllm` | 依赖未安装 | `pip install --force-reinstall -r requirements_batch.txt` |

### 性能基准

| GPU型号 | 处理速度 | 显存使用 | 推荐配置 | 备注 |
|---------|----------|----------|----------|------|
| **RTX 4090 48G** | **40-90秒/PDF** | **38-43GB** | **BATCH_SIZE=12, MAX_CONCURRENCY=16** | **当前优化目标** |
| RTX 3090 24G | 2-5分钟/PDF | 18-22GB | BATCH_SIZE=4, MAX_CONCURRENCY=6 | 原配置 |
| RTX 4090 24G | 1-3分钟/PDF | 16-20GB | BATCH_SIZE=6, MAX_CONCURRENCY=8 | - |
| A100 40G | 1-2分钟/PDF | 15-18GB | BATCH_SIZE=8, MAX_CONCURRENCY=12 | - |
| A100 80G | 30-60秒/PDF | 30-40GB | BATCH_SIZE=16, MAX_CONCURRENCY=20 | 最强配置 |

**性能提升对比（RTX 4090 48G vs RTX 3090 24G）：**
- 小文档 (5-10页): **3-4倍** 提升
- 中文档 (20-30页): **3-4倍** 提升
- 大文档 (50+页): **3-4倍** 提升
- 批量处理: **3-4倍** 提升

### 日志分析

**关键日志位置：**
- `logs/batch_processor.log` - 主处理日志
- `logs/errors.log` - 错误日志
- `output_results/{filename}/processing_log.txt` - 单文件处理日志

**常见错误模式：**
```bash
# 查找API错误
grep "API调用失败" logs/batch_processor.log

# 查找显存错误
grep "CUDA out of memory" logs/errors.log

# 查找验证错误
grep "Schema验证失败" logs/batch_processor.log
```

## 环境变量

必需的环境变量（可在 `.env` 文件中配置）：

```bash
# OpenRouter API密钥（必需）
OPENROUTER_API_KEY=sk-or-v1-xxxxx

# DeepSeek OCR模型路径（可选，默认从HuggingFace下载）
DEEPSEEK_OCR_MODEL_PATH=deepseek-ai/DeepSeek-OCR

# CUDA设备选择（可选，默认0）
CUDA_VISIBLE_DEVICES=0

# vLLM版本控制（可选，默认0）
VLLM_USE_V1=0

# CUDA 11.8特定配置（可选）
TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
```

## 代码风格

- **Python版本：** 3.10+
- **类型注解：** 使用 `typing` 模块的类型提示
- **异步编程：** OpenRouter API调用使用 `asyncio` + `AsyncOpenAI`
- **错误处理：** 使用 `try-except` 并记录详细日志
- **配置管理：** 使用 `dataclass` 和配置类
- **路径处理：** 统一使用 `pathlib.Path`

## 测试

运行完整测试套件：
```bash
python test_batch_system.py
```

测试覆盖：
- 环境配置验证
- GPU显存检查
- 模型加载测试
- JSON Schema验证
- 图像处理测试
- API连接测试

## 项目文件说明

**核心文件：**
- `batch_pdf_processor.py` - 主处理器（RTX 4090优化版）
- `config_batch.py` - 配置管理（RTX 4090极速配置）
- `deepseek_ocr.py` - OCR模型实现（vLLM集成）
- `run_batch_processor.py` - 启动脚本
- `test_batch_system.py` - 系统测试

**配置文件：**
- `config.py` - 基础OCR配置
- `json schema.json` - 输出数据Schema v1.3.1
- `requirements_batch.txt` - Python依赖

**文档：**
- `CLAUDE.md` - 本文件（开发指南）
- `RTX4090_OPTIMIZATION.md` - RTX 4090优化详细说明
- `README.md` - 项目说明

**辅助模块：**
- `process/image_process.py` - 图像预处理
- `process/ngram_norepeat.py` - 重复检测
- `deepencoder/` - 视觉编码器（SAM + CLIP）

**目录结构（优化后）：**
- `input_pdfs/` - PDF输入目录
- `output_results/` - OCR结果输出（仅MD和图像）
- `output_report/` - JSON报告输出（新增，专门存放JSON）
- `temp_processing/` - 临时文件
- `logs/` - 日志文件

## 注意事项

1. **显存管理：** RTX 4090 48G 显存充足，可同时运行多个处理任务，但建议监控显存使用
2. **API配额：** OpenRouter API有速率限制，当前配置12并发，确保API配额充足
3. **Schema验证：** 输出JSON必须严格符合v1.3.1 Schema，已启用严格验证模式
4. **路径问题：** 已修复路径识别问题，支持符号链接和复杂目录结构
5. **环境变量：** 确保 `OPENROUTER_API_KEY` 已正确设置
6. **模型下载：** 首次运行会从HuggingFace下载模型（约10GB），需要稳定网络
7. **CUDA版本：** 项目针对CUDA 11.8优化，其他版本可能需要调整PyTorch安装命令
8. **输出目录：** 注意新的目录结构，OCR结果在 `output_results/`，JSON在 `output_report/`
9. **并发控制：** 高并发可能导致API限流，根据实际情况调整配置
10. **网络稳定性：** 12并发API调用需要稳定的网络连接

## RTX 4090 优化详情

详细的优化说明请参考 `RTX4090_OPTIMIZATION.md` 文档，包括：
- 完整的配置对比
- 性能测试结果
- 调优建议
- 监控方法
- 故障排除
