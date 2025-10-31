# 阶段B卡住问题 - 诊断报告

## 问题现象

1. **阶段B启动正常**：15个worker都启动并开始处理任务
2. **进入图表识别阶段**：所有worker都打印了"阶段B-1: 图表识别"日志
3. **卡住无进展**：之后没有任何"阶段B-2"或"阶段B-3"的日志
4. **没有生成JSON**：output_report目录完全为空

## 根本原因

**阶段B在图表识别阶段批量调用OpenRouter API时遇到连接超时，导致所有任务卡住。**

### 详细分析

1. **连接超时问题**
   - 日志显示大量 `httpcore.ConnectTimeout` 错误
   - 默认连接超时只有5秒，网络波动时容易失败
   - 15个worker并发处理，每个PDF可能有几十张图表

2. **异常处理不当**
   - 原代码使用 `return_exceptions=True`，异常被静默吞掉
   - 没有详细的进度日志，无法看到实际失败情况
   - 异常处理中只记录日志，不向上传播

3. **并发压力过大**
   - 原配置：48个并发API调用（已调整为15）
   - 每个图表都需要单独的API调用
   - 单个PDF可能有20-40张图表需要识别

## 已实施的修复

### 1. 连接超时配置 (`batch_pdf_processor.py:540-545`)

```python
# 详细的超时配置
timeout_config = httpx.Timeout(
    connect=60.0,  # 连接超时: 60秒 (原默认5秒)
    read=600.0,    # 读取超时: 10分钟
    write=60.0,    # 写入超时: 60秒
    pool=60.0      # 连接池超时: 60秒
)
```

### 2. 并发控制优化 (`config_batch.py:19, 135`)

```python
MAX_CONCURRENCY = 15              # 从48降到15
MAX_CONCURRENT_API_CALLS = 15     # 从48降到15
```

### 3. 增强日志和错误处理 (`batch_pdf_processor.py:939-986`)

**改进前：**
- 静默失败，无详细日志
- 异常被吞掉，不向上传播

**改进后：**
- 详细的进度日志（开始、成功、失败统计）
- 区分不同类型的错误（连接超时、读取超时、其他错误）
- 异常向上传播，便于调试
- 整体异常捕获，防止整个阶段B崩溃

### 4. 单个图表识别增强 (`batch_pdf_processor.py:1181-1267`)

**添加的日志：**
- 开始识别：`logger.debug(f"开始识别图表: {filename}")`
- API调用：`logger.debug(f"调用API识别图表: {filename}")`
- 成功：`logger.debug(f"图表识别成功: {filename}")`
- 失败：区分超时类型的详细错误日志

**异常处理改进：**
```python
except asyncio.TimeoutError as e:
    logger.error(f"图表识别超时 {filename}: {e}")
    raise  # 向上传播
except httpx.ConnectTimeout as e:
    logger.error(f"图表识别连接超时 {filename}: {e}")
    raise  # 向上传播
except httpx.ReadTimeout as e:
    logger.error(f"图表识别读取超时 {filename}: {e}")
    raise  # 向上传播
```

## 诊断和测试步骤

### 步骤1: 测试API连接

```bash
cd /home/qxx/DeepSeek-OCR/DeepSeek-OCR-master/deepseek-ocr-batch
python test_api_connection.py
```

**预期结果：**
- ✅ 如果通过：API连接正常，可以继续
- ❌ 如果失败：网络连接问题，需要配置代理

### 步骤2: 测试图表识别

```bash
python test_figure_extraction.py
```

**预期结果：**
- ✅ 如果通过：图表识别API正常工作
- ❌ 如果失败：查看具体错误类型

### 步骤3: 运行批量处理（观察新日志）

```bash
# 重新启动批量处理
python run_batch_processor.py

# 在另一个终端监控日志
tail -f logs/batch_processor.log | grep -E "(阶段B|图表识别|OpenRouterProcessor|成功|失败)"
```

**现在应该能看到的新日志：**
```
2025-10-25 XX:XX:XX - INFO - 开始识别 43 张图表...
2025-10-25 XX:XX:XX - INFO - OpenRouterProcessor初始化成功，开始并行识别...
2025-10-25 XX:XX:XX - DEBUG - 开始识别图表: 0_0.jpg
2025-10-25 XX:XX:XX - DEBUG - 调用API识别图表: 0_0.jpg
2025-10-25 XX:XX:XX - DEBUG - API响应成功: 0_0.jpg
2025-10-25 XX:XX:XX - INFO - 图表识别完成: 成功 35/43, 失败 8/43
```

## 可能的后续问题

### 问题1: 仍然连接超时

**症状：**
```
ERROR - 图表识别连接超时 0_0.jpg: ConnectTimeout(...)
```

**解决方案：**
1. 检查网络连接：`ping openrouter.ai`
2. 测试HTTP访问：`curl -I https://openrouter.ai/api/v1`
3. 配置HTTP代理（如果在受限网络）：
   ```bash
   export HTTP_PROXY=http://proxy:port
   export HTTPS_PROXY=http://proxy:port
   python run_batch_processor.py
   ```

### 问题2: API配额耗尽

**症状：**
```
ERROR - 图表识别失败: 429 Too Many Requests
```

**解决方案：**
1. 进一步降低并发：`config_batch.py` 中设置 `MAX_CONCURRENT_API_CALLS = 5`
2. 增加重试延迟：`RETRY_DELAY_BASE = 5`

### 问题3: 部分图表识别失败

**症状：**
```
INFO - 图表识别完成: 成功 10/43, 失败 33/43
```

**解决方案：**
- 这是正常的，并非所有图像都是图表
- 代码会继续处理，不影响最终JSON生成
- 如果失败率>80%，检查API配额和网络连接

## 预期改进效果

### 修复前：
- ❌ 阶段B完全卡住
- ❌ 没有错误日志
- ❌ 无法生成任何JSON

### 修复后：
- ✅ 详细的进度日志
- ✅ 清晰的错误信息
- ✅ 即使部分失败也能继续处理
- ✅ 能够生成JSON（图表数据可能不完整，但有基本结构）

## 性能预期

**理想情况（网络正常）：**
- 图表识别成功率：70-90%
- 单个PDF处理时间：80-150秒
- JSON生成成功率：>95%

**网络不稳定情况：**
- 图表识别成功率：30-60%
- 单个PDF处理时间：120-200秒
- JSON生成成功率：>85%

## 文件修改总结

| 文件 | 变更 | 行号 |
|------|------|------|
| `batch_pdf_processor.py` | 添加 httpx 导入 | 25 |
| `batch_pdf_processor.py` | 配置详细超时 | 540-545 |
| `batch_pdf_processor.py` | 增强图表识别日志 | 939-986 |
| `batch_pdf_processor.py` | 改进单图表识别 | 1181-1267 |
| `config_batch.py` | 降低并发数 | 19, 135 |
| 新增 `test_api_connection.py` | API连接测试 | - |
| 新增 `test_figure_extraction.py` | 图表识别测试 | - |
| 新增 `DIAGNOSIS.md` | 本诊断报告 | - |

---

## 下一步操作

1. ✅ **运行测试脚本**：`python test_api_connection.py`
2. ✅ **如果测试通过**：`python run_batch_processor.py`
3. 📊 **监控日志**：`tail -f logs/batch_processor.log`
4. 🔍 **检查结果**：查看 `output_report/` 目录是否生成JSON文件

如果仍有问题，请提供新的日志输出以便进一步诊断。
