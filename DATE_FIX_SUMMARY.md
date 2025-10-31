# 日期识别错误修复总结

## 📋 问题描述

**发现的问题：**
- `output_report/` 和 `output_results/` 中存在日期为 `2025-12-31` 的文件夹
- 这些日期是错误的（当前为2025年10月，12月31日还未到达）
- 问题源于 PDF 文件名本身带有错误的日期后缀

## 🔍 原因分析

**根本原因：**
PDF 文件在命名时，日期后缀被错误标记为 `_2025-12-31.pdf`，导致系统从文件名中提取日期时使用了错误的值。

**示例：**
```
错误的PDF文件名：
Bank of Beijing - A 2Q25 Mixed Profits_2025-12-31.pdf

实际应该是：
Bank of Beijing - A 2Q25 Mixed Profits_2025-09-03.pdf
```

## ✅ 修复方案

创建了两个修复脚本：

### 1. `fix_wrong_dates.py` - 修复 output_report

**功能：**
- 扫描 `output_report/` 中错误日期的 JSON 文件
- 从对应 PDF 的文件修改时间中提取正确日期
- 将文件移动到正确的日期目录

**使用方法：**
```bash
python fix_wrong_dates.py --yes
```

### 2. `fix_output_results_dates.py` - 修复 output_results

**功能：**
- 扫描 `output_results/` 中错误日期的文件夹
- 从对应 PDF 的文件修改时间中提取正确日期
- 将整个文件夹移动到正确的日期目录

**使用方法：**
```bash
python fix_output_results_dates.py --yes
```

## 📊 修复结果

### output_report 修复

**修复前：**
```
output_report/2025-12-31/
├── Bank of Beijing...json (8个文件)
├── Bank of Hangzhou...json
├── ...
```

**修复后：**
```
output_report/2025-09-02/
├── Bank of Hangzhou...json (4个文件)
├── China Taiping...json
└── ...

output_report/2025-09-03/
├── Bank of Beijing...json (4个文件)
├── Bank of Nanjing...json
└── ...

✅ 2025-12-31/ 目录已删除
```

**统计：**
- 修复文件数：8个 JSON 文件
- 失败数：0
- 新日期分布：
  - 2025-09-02: 4个文件
  - 2025-09-03: 4个文件

### output_results 修复

**修复前：**
```
output_results/2025-12-31/
├── Bank of Beijing.../  (6个文件夹)
├── Bank of Hangzhou.../
├── ...
```

**修复后：**
```
output_results/2025-09-02/
├── Bank of Hangzhou.../ (2个文件夹)
├── China Taiping.../
└── ...

output_results/2025-09-03/
├── Bank of Beijing.../ (4个文件夹)
├── Bank of Nanjing.../
└── ...

✅ 2025-12-31/ 目录已删除
```

**统计：**
- 修复文件夹数：6个
- 失败数：0
- 新日期分布：
  - 2025-09-02: 2个文件夹
  - 2025-09-03: 4个文件夹

## 🔧 技术细节

### 日期提取逻辑

1. **优先级1：PDF元数据** (如果有 PyMuPDF)
   - 从 PDF 的 `creationDate` 或 `modDate` 字段提取
   - 格式：`D:20250903...` → `2025-09-03`

2. **优先级2：文件修改时间** (fallback)
   - 使用 `os.path.getmtime()` 获取文件修改时间
   - 转换为 `YYYY-MM-DD` 格式

3. **验证规则：**
   - 只接受 2025 年 1-10 月的日期
   - 超出范围的日期视为无效

### 修复流程

```
1. 扫描错误日期目录
   ↓
2. 遍历每个JSON/文件夹
   ↓
3. 在 input_pdfs/ 中查找对应PDF
   ↓
4. 从PDF提取正确日期
   ↓
5. 创建正确日期目录
   ↓
6. 移动文件/文件夹
   ↓
7. 删除空的错误日期目录
```

## 📝 执行记录

### 第一步：修复 output_report

```bash
$ python fix_wrong_dates.py --yes

🔍 第一步：检查需要修复的文件（DRY RUN）
📁 处理目录: 2025-12-31 (8 个JSON文件)
  ✅ 找到对应PDF，提取日期
  🔄 将移动到正确目录

🔧 第二步：执行修复...
✅ 修复 8 个文件，失败 0 个
🗑️ 已删除空目录: 2025-12-31
```

### 第二步：修复 output_results

```bash
$ python fix_output_results_dates.py --yes

🔍 第一步：检查需要修复的文件夹（DRY RUN）
📁 处理目录: 2025-12-31 (6 个文件夹)
  ✅ 找到对应PDF，提取日期
  🔄 将移动到正确目录

🔧 第二步：执行修复...
✅ 修复 6 个文件夹，失败 0 个
🗑️ 已删除空目录: 2025-12-31
```

## ✅ 验证结果

```bash
# 验证 output_report
$ find output_report -name "2025-12-31" -type d
# (无输出 - 目录已删除)

# 验证 output_results
$ find output_results -name "2025-12-31" -type d
# (无输出 - 目录已删除)

# 确认文件已移动
$ ls output_report/2025-09-02/*.json | wc -l
106  # 包含新移动的4个文件

$ ls output_report/2025-09-03/*.json | wc -l
4    # 新移动的4个文件
```

## 🎯 后续建议

### 1. 修复 input_pdfs 中的PDF文件名

**问题：** PDF 文件名本身带有错误日期后缀

**建议：** 创建批量重命名脚本
```bash
# 示例
mv "Bank of Beijing_2025-12-31.pdf" "Bank of Beijing_2025-09-03.pdf"
```

### 2. 改进日期提取逻辑

**当前问题：** 系统从文件名提取日期，如果文件名错误则结果也错误

**改进方案：**
- 优先从 PDF 元数据提取日期
- 添加日期合理性检查（不接受未来日期）
- 在处理时记录警告日志

**代码位置：** `batch_pdf_processor.py` 中的日期提取函数

### 3. 添加日期验证到处理流程

在 `batch_pdf_processor.py` 中添加验证：
```python
def validate_date(date_str: str) -> bool:
    """验证日期是否合理"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        now = datetime.now()
        # 不接受未来日期
        if dt > now:
            return False
        # 不接受过老的日期（比如2020年之前）
        if dt.year < 2020:
            return False
        return True
    except ValueError:
        return False
```

## 📂 相关文件

**修复脚本：**
- `fix_wrong_dates.py` - output_report 修复脚本
- `fix_output_results_dates.py` - output_results 修复脚本

**影响的目录：**
- `output_report/2025-12-31/` → 已删除
- `output_results/2025-12-31/` → 已删除
- `output_report/2025-09-02/` ← 新增4个文件
- `output_report/2025-09-03/` ← 新增4个文件
- `output_results/2025-09-02/` ← 新增2个文件夹
- `output_results/2025-09-03/` ← 新增4个文件夹

## 🔒 安全措施

**脚本特性：**
1. ✅ Dry-run 模式：先预览再执行
2. ✅ 目标文件存在检查：避免覆盖
3. ✅ 错误处理：记录失败的文件
4. ✅ 自动删除空目录：保持目录整洁
5. ✅ 详细日志：每个操作都有记录

**执行建议：**
- 首次运行先不加 `--yes` 参数，查看 dry-run 结果
- 确认无误后再加 `--yes` 自动执行
- 重要数据先备份

---

**修复日期：** 2025-10-27
**修复者：** Claude Code
**问题报告者：** 用户
**状态：** ✅ 已完成
**影响文件数：** 14个（8个JSON + 6个文件夹）
