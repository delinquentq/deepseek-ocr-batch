# 进度显示修复说明

## 问题描述

**现象：**
- 日志显示"阶段A: 跳过 841 个已生成MD的文件，待处理 113 个"
- 但进度显示为"[阶段A 1/954]"而不是"[阶段A 1/113]"
- 用户误以为系统在重新处理所有954个文件

**实际情况：**
- 系统**正确跳过了**841个已有MD的文件
- 只处理113个待处理文件
- 问题仅在于**进度显示的分母用错了**

## 根本原因

在 `batch_pdf_processor.py:2713` 行：

```python
# 修复前（错误）
total = len(pdf_paths)  # 使用所有文件数954

async def stage_a_worker(pdf_path: str, index: int):
    logger.info(f"[阶段A {index}/{total}] ...")  # 显示 1/954
```

**问题：**
- `total` 被设置为所有文件数（954）
- 但实际只处理 `stage_a_pending`（113个文件）
- 导致进度显示 "1/954" 而非 "1/113"

## 修复方案

### 代码修改

**batch_pdf_processor.py:2713-2715**
```python
# 修复后
total_pending = len(stage_a_pending)  # 待处理文件数：113
total_all = len(pdf_paths)            # 所有文件数：954
```

**batch_pdf_processor.py:2741, 2749, 2757**
```python
# 修复后：使用 total_pending
f"[阶段A {index}/{total_pending}] 准备: {pdf_name}"
f"[阶段A {index}/{total_pending}] 入队完成: {pdf_name}"
f"[阶段A {index}/{total_pending}] 失败: {pdf_name}"
```

### 修复效果

**修复前：**
```
阶段A: 跳过 841 个已生成MD的文件，待处理 113 个
[阶段A 1/954] 准备: xxx.pdf  ❌ 误导性显示
[阶段A 2/954] 准备: yyy.pdf
...
```

**修复后：**
```
阶段A: 跳过 841 个已生成MD的文件，待处理 113 个
[阶段A 1/113] 准备: xxx.pdf  ✅ 正确显示
[阶段A 2/113] 准备: yyy.pdf
...
[阶段A 113/113] 入队完成: zzz.pdf  ✅ 进度清晰
```

## 验证方法

### 1. 运行测试

```bash
# 重新运行批量处理
python run_batch_processor.py -y --input "input_pdfs/25.9月 普通个人版/9.2 普通个人版"

# 观察日志
tail -f logs/batch_processor.log | grep "阶段A"
```

### 2. 预期输出

```
阶段A: 跳过 841 个已生成MD的文件，待处理 113 个
[阶段A 1/113] 准备: xxx.pdf      ✅ 分母为113
[阶段A 2/113] 入队完成: xxx.pdf
[阶段A 3/113] 准备: yyy.pdf
...
[阶段A 113/113] 入队完成: zzz.pdf  ✅ 最后一个
```

## 技术细节

### 变量说明

| 变量 | 含义 | 示例值 |
|------|------|--------|
| `pdf_paths` | 所有输入的PDF文件列表 | 954个 |
| `stage_a_pending` | 需要处理的PDF（无MD）| 113个 |
| `stage_a_skipped` | 已跳过的PDF（有MD）| 841个 |
| `total_pending` | 待处理数（新增）| 113 |
| `total_all` | 总文件数（新增）| 954 |

### 修改文件

- **文件：** `batch_pdf_processor.py`
- **修改行：** 2713-2715, 2741, 2749, 2757
- **修改类型：** Bug修复（进度显示逻辑）
- **向后兼容：** 是（不影响处理逻辑）

## 相关代码位置

```python
# batch_pdf_processor.py

async def process_batch(self, pdf_paths: List[str]) -> List[Dict]:
    """批量处理PDF文件，采用阶段A/B流水线并发模式"""
    total = len(pdf_paths)

    # 过滤已完成的文件
    stage_a_pending = [p for p in pdf_paths if not self._is_stage_a_completed(p)]
    stage_a_skipped = total - len(stage_a_pending)

    logger.info(f"阶段A: 跳过 {stage_a_skipped} 个已生成MD的文件，待处理 {len(stage_a_pending)} 个")

    # 🔥 修复：使用待处理文件数
    total_pending = len(stage_a_pending)  # ← 新增
    total_all = len(pdf_paths)            # ← 新增

    async def stage_a_worker(pdf_path: str, index: int):
        logger.info(f"[阶段A {index}/{total_pending}] ...")  # ← 修复
```

## 影响范围

### ✅ 修复内容
- 阶段A进度显示分母现在正确显示待处理文件数
- 用户可以清楚看到实际处理进度（1/113 而非 1/954）

### 🔄 不影响的内容
- 阶段A跳过逻辑（正常工作）
- 阶段B处理逻辑（正常工作）
- 并发控制逻辑（正常工作）
- JSON输出结果（不受影响）

## 测试清单

- [x] Python语法验证通过
- [x] 变量命名正确
- [x] 进度显示逻辑修复
- [ ] 实际运行测试（待用户确认）
- [ ] 完整批处理验证（待用户确认）

## 总结

这是一个**视觉显示问题**，不是功能问题：
- ✅ 系统一直都正确跳过已处理文件
- ✅ 系统一直都只处理待处理文件
- ❌ 只是进度显示误导了用户
- ✅ 现在进度显示准确反映实际处理情况

---

**修复日期：** 2025-10-26
**修复者：** Claude Code
**问题报告者：** 用户
**严重程度：** 低（仅影响显示，不影响功能）
**状态：** 已修复，待测试
