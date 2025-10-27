# 阶段B无法生成JSON - 完整诊断报告

## 问题现状

**症状：**
- ✅ 阶段A（OCR）正常工作，生成MD和图像
- ✅ 阶段B-1（图表识别）能够完成
- ❌ 阶段B-2（文本提取）无法完成
- ❌ output_report/ 目录完全为空，没有生成任何JSON

**最新日志显示：**
```
connect_tcp.failed exception=CancelledError()
```
大量的连接取消错误，说明**异步任务被提前取消**。

---

## 根本原因分析

### 1. **CancelledError 的原因**

`CancelledError` 通常由以下原因引起：

**原因A: 并发冲突**
- 15个阶段B worker同时调用API
- 每个worker在图表识别后又创建OpenRouterProcessor
- 多个AsyncOpenAI客户端同时初始化时竞争连接
- 导致部分连接被取消

**原因B: 超时导致的级联取消**
- 某个API调用超时
- `asyncio.gather()` 检测到异常
- 取消所有其他pending的任务
- 导致大量CancelledError

**原因C: 资源竞争**
- 15个worker × 多个API调用 = 大量并发连接
- 超过系统或OpenRouter的连接限制
- 新连接被拒绝或取消

### 2. **为什么没有JSON生成**

查看 `batch_pdf_processor.py:2649`：
```python
result = await self._process_pdf_stage_b(job)
```

如果 `_process_pdf_stage_b` 抛出异常，会被 `try-except` 捕获（2655行），但**只记录错误，不生成JSON**。

---

## 解决方案

### **方案1: 降低并发数（最直接）**

```bash
# 编辑 config_batch.py
```

```python
class ProcessingConfig:
    # 修改前
    MAX_CONCURRENT_PDFS = 6
    MAX_CONCURRENT_API_CALLS = 15

    # 修改后（大幅降低）
    MAX_CONCURRENT_PDFS = 2        # 从6降到2
    MAX_CONCURRENT_API_CALLS = 5    # 从15降到5
```

**理由：** 减少同时运行的worker和API调用，避免连接竞争。

---

### **方案2: 复用OpenRouterProcessor（推荐）**

**问题：** 每个阶段B-2都创建新的OpenRouterProcessor，导致连接过多。

**修改 `batch_pdf_processor.py:1284`：**

```python
# 修改前（每次都创建新的processor）
async def _process_with_single_model_simplified(...):
    async with OpenRouterProcessor() as processor:
        result = await self._call_model_with_prompt(...)

# 修改后（接收已有的processor）
async def _process_with_single_model_simplified(
    self,
    markdown_content: str,
    pdf_name: str,
    page_count: int,
    date_str: str,
    publication: str,
    figures_data: List[Dict],
    processor: OpenRouterProcessor  # 新增参数
) -> Dict:
    """简化的单模型处理（整合文本和图表数据）"""
    logger.debug(f"开始单模型处理: {pdf_name}")
    results: Dict[str, Any] = {}

    try:
        extraction_prompt = self._build_simplified_extraction_prompt(
            markdown_content, pdf_name, page_count, date_str, publication, figures_data
        )
        logger.debug(f"提示词构建完成，长度: {len(extraction_prompt)} 字符")

        # 直接使用传入的processor，不再创建新的
        logger.debug(f"使用已有的processor调用模型...")
        try:
            result = await self._call_model_with_prompt(
                processor, "gemini", extraction_prompt, pdf_name
            )
            logger.debug(f"模型调用成功")
            if result["result"]:
                if "data" not in result["result"]:
                    result["result"]["data"] = {}
                result["result"]["data"]["figures"] = figures_data
            results["gemini"] = result["result"]
            logger.debug(f"单模型处理完成")
        except Exception as e:
            logger.error(f"gemini模型调用失败: {type(e).__name__}: {e}")
            results["gemini"] = {}
    except Exception as e:
        logger.error(f"单模型处理整体失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        results["gemini"] = {}

    return results
```

**然后修改调用处 `batch_pdf_processor.py:837-848`：**

```python
logger.info(f"{Colors.BLUE}阶段B-1: 图表识别{Colors.RESET}")
# 创建一个processor用于整个阶段B
async with OpenRouterProcessor() as processor:
    figures_data = await self._extract_figures_data_parallel_with_processor(
        job.figure_paths, processor
    )
    logger.info(f"{Colors.CYAN}阶段B-1完成: 识别 {len(figures_data)} 张图表{Colors.RESET}")

    logger.info(f"{Colors.BLUE}阶段B-2: 文本与图表联合提取{Colors.RESET}")
    logger.debug(f"开始阶段B-2: 调用单模型处理...")
    model_results = await self._process_with_single_model_simplified(
        job.markdown_content,
        pdf_name,
        page_count,
        job.date_str or "",
        job.publication,
        figures_data,
        processor  # 传入processor
    )
    # ... 后续处理 ...
```

---

### **方案3: 增加重试机制**

在 `consumer_worker` 中增加重试：

```python
async def consumer_worker(worker_id: int) -> None:
    logger.info(f"[阶段B#{worker_id}] Worker已启动，等待任务...")
    while True:
        try:
            job = await job_queue.get()
            if job is None:
                job_queue.task_done()
                break

            pdf_name = Path(job.pdf_path).name
            queue_size = job_queue.qsize()
            logger.info(f"[阶段B#{worker_id}] 开始处理: {pdf_name} (队列剩余: {queue_size})")

            # 增加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = await self._process_pdf_stage_b(job)
                    results.append(result)
                    logger.info(f"[阶段B#{worker_id}] ✓ 完成: {pdf_name}")
                    break  # 成功则跳出重试循环
                except asyncio.CancelledError:
                    if attempt < max_retries - 1:
                        logger.warning(f"[阶段B#{worker_id}] 任务被取消，重试 {attempt+1}/{max_retries}: {pdf_name}")
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                        continue
                    else:
                        raise  # 最后一次重试失败，向上抛出
                except Exception as exc:
                    if attempt < max_retries - 1:
                        logger.warning(f"[阶段B#{worker_id}] 处理失败，重试 {attempt+1}/{max_retries}: {pdf_name} - {exc}")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        failed_files.add(job.pdf_path)
                        logger.error(f"[阶段B#{worker_id}] ✗ 失败: {pdf_name} - {exc}")
                        logger.error(f"完整错误堆栈:\\n{traceback.format_exc()}")
                        break
        finally:
            job_queue.task_done()
```

---

## 推荐执行步骤

### **步骤1: 快速修复（降低并发）**

```bash
cd /home/qxx/DeepSeek-OCR/DeepSeek-OCR-master/deepseek-ocr-batch

# 编辑配置文件
nano config_batch.py

# 修改以下参数:
# MAX_CONCURRENT_PDFS = 2
# MAX_CONCURRENT_API_CALLS = 5

# 重新运行
python run_batch_processor.py
```

### **步骤2: 监控效果**

```bash
# 在另一个终端
tail -f logs/batch_processor.log | grep --color -E "阶段B|完成|失败|ERROR|CancelledError"
```

### **步骤3: 检查JSON生成**

```bash
# 等待5-10分钟后检查
ls -lh output_report/2025-09-*/

# 如果有JSON文件生成
ls output_report/2025-09-28/*.json | head -5
```

---

## 预期结果

**修复后应该看到：**
```
INFO - [阶段B#1] 开始处理: Aerospace...
INFO - 阶段B: API推理与结构化处理
INFO - 阶段B-1: 图表识别
INFO - 开始识别 43 张图表...
INFO - 图表识别完成: 成功 35/43, 失败 8/43
INFO - ✓ 阶段B-1完成: 识别 35 张图表
INFO - 阶段B-2: 文本与图表联合提取
DEBUG - 开始单模型处理: Aerospace...
DEBUG - 模型调用成功
INFO - ✓ 阶段B-2完成: 文本与图表数据已提取
INFO - 阶段B-3: 基础验证与补全
INFO - ✓ 保存JSON: output_report/2025-09-28/Aerospace....json  ← 关键
INFO - [阶段B#1] ✓ 完成: Aerospace... (耗时: 180.5秒)
```

**不应该再看到：**
```
❌ connect_tcp.failed exception=CancelledError()
```

---

## 如果仍然失败

请提供以下信息：

1. **新的错误日志：**
   ```bash
   grep -E "ERROR|失败|CancelledError" logs/batch_processor.log | tail -30
   ```

2. **阶段B完成情况：**
   ```bash
   grep "阶段B.*完成" logs/batch_processor.log | wc -l
   ```

3. **JSON文件数量：**
   ```bash
   find output_report/ -name "*.json" | wc -l
   ```

---

**立即执行：降低并发数，然后重新运行！**
