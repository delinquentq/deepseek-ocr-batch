# 错误日期PDF文件清单

## 📊 统计总览

| 错误日期 | PDF数量 | 状态 |
|---------|---------|------|
| 2025-12-31 | 7个 | ✅ 输出已修复 |
| 2025-12-09 | 1个 | ℹ️ 仅此一个 |
| 2025-12-04 | 1个 | ℹ️ 仅此一个 |
| 2025-11-10 | 1个 | ℹ️ 仅此一个 |
| **总计** | **10个** | |

## 📁 详细清单

### 1. 日期 2025-12-31（7个文件）

**位置：** `input_pdfs/25.9月 普通个人版/`

#### 9.2 普通个人版/JPM/ (4个)
```
1. Bank of Beijing - A 2Q25 Mixed Profits beat while NIM missed. Capital position may restrict the stability of dividend po_2025-12-31.pdf
   实际日期: 2025-09-03

2. Bank of Nanjing - A 2Q25 - Strong PPOP growth beat expectationBank of Nanjing - A 2Q25 - Strong PPOP growth beat expecta_2025-12-31.pdf
   实际日期: 2025-09-03

3. Bank of Ningbo - A 2Q25- Core earnings growth back on trackBank of Ningbo - A 2Q25- Core earnings growth back on track_2025-12-31.pdf
   实际日期: 2025-09-03

4. Longfor Group 0960 1H25 growth in recurring income slower than expected but the balance sheet improvesLongfor Group 0960_2025-12-31.pdf
   实际日期: 2025-09-03
```

#### 9.1 普通个人版/JPM/ (2个)
```
5. Bank of Hangzhou - A 2Q25 Strong PPOP and profits growth while NIM pressure lingersBank of Hangzhou - A 2Q25 Strong PPOP_2025-12-31.pdf
   实际日期: 2025-09-02

6. China Taiping Insurance - H First Take 1H25 results fall short three reasons for caution following a strong YTD rallyChi_2025-12-31.pdf
   实际日期: 2025-09-02
```

**位置：** `input_pdfs/25.10月 普通个人版/`

#### 10.9 普通个人版/JPM/ (1个)
```
7. Geely Automobile Holdings Ltd. 0175 Share buyback reveals mgmt confidence in the companys futureGeely Automobile Holding_2025-12-31.pdf
   实际日期: 2025-10-09
```

### 2. 日期 2025-12-09（1个文件）

**位置：** `input_pdfs/25.9月 普通个人版/9.26 普通个人版/DB/`

```
Fed SOMA Holdings Analytics for Treasury securities in SOMAFed SOMA Holdings Analytics for Treasury securities in SOMA(3)_2025-12-09.pdf
   实际日期: 2025-09-26
```

### 3. 日期 2025-12-04（1个文件）

**位置：** `input_pdfs/25.9月 普通个人版/9.16 普通个人版/Jefferies/`

```
Monthly Barometer ve EDPELE small -ve SSE August 2025 DataMonthly Barometer ve EDPELE small -ve SSE August 2025 Data_2025-12-04.pdf
   实际日期: 2025-09-16
```

### 4. 日期 2025-11-10（1个文件）

**位置：** `input_pdfs/25.10月 普通个人版/10.13 普通个人版/GS/`

```
Sodexo EXHO.PA New external CEO appointedSodexo EXHO.PA New external CEO appointed_2025-11-10.pdf
   实际日期: 2025-10-13
```

## 🔍 问题分析

### 日期错误的来源

这些PDF文件的错误日期**不是系统处理产生的**，而是：

1. **上游命名错误**：PDF文件在下载/命名时就带有错误的日期后缀
2. **可能的原因**：
   - 自动命名脚本使用了错误的日期变量
   - 批量重命名时设置了错误的默认日期
   - 复制粘贴时继承了错误的日期

### 日期规律观察

| 错误日期 | 实际应该是 | 差异 | 特点 |
|---------|-----------|------|------|
| 2025-12-31 | 2025-09-02/03/10-09 | 跳到年末 | **最常见**（7个） |
| 2025-12-09 | 2025-09-26 | +74天 | |
| 2025-12-04 | 2025-09-16 | +79天 | |
| 2025-11-10 | 2025-10-13 | +28天 | |

**观察：**
- 2025-12-31 是最常见的错误，可能是某个脚本的默认值
- 所有错误日期都是"未来日期"
- 主要集中在 JPM 机构的文件

## ✅ 已完成的修复

### 修复范围

我已经修复了**输出文件**的日期：

1. ✅ `output_report/` - 8个JSON文件从 2025-12-31 移到正确日期
2. ✅ `output_results/` - 6个文件夹从 2025-12-31 移到正确日期

### 未修复的部分

⚠️ **源PDF文件名未修改**（保持原样）

**原因：**
- 避免破坏原始数据
- 不确定是否有其他系统依赖这些文件名
- 用户可能需要保留原始命名

## 🛠️ 如何修复源PDF文件名（可选）

如果你想重命名源PDF文件，可以使用这个脚本：

```python
# rename_pdfs.py
from pathlib import Path
import shutil
import os
from datetime import datetime

def rename_wrong_date_pdfs():
    """重命名带错误日期的PDF文件"""

    wrong_dates = {
        "2025-12-31": None,  # 需要从文件时间推断
        "2025-12-09": None,
        "2025-12-04": None,
        "2025-11-10": None,
    }

    for wrong_date in wrong_dates:
        pattern = f"*_{wrong_date}.pdf"
        pdfs = list(Path("input_pdfs").rglob(pattern))

        for pdf in pdfs:
            # 从文件修改时间获取正确日期
            mtime = os.path.getmtime(pdf)
            correct_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

            # 新文件名
            new_name = pdf.name.replace(f"_{wrong_date}.pdf", f"_{correct_date}.pdf")
            new_path = pdf.parent / new_name

            print(f"重命名: {pdf.name}")
            print(f"  → {new_name}")

            # 取消下面的注释来实际执行重命名
            # shutil.move(str(pdf), str(new_path))

if __name__ == "__main__":
    rename_wrong_date_pdfs()
```

## 📌 建议

### 短期建议

1. ✅ **输出文件已修复** - 无需额外操作
2. ℹ️ **源PDF文件** - 根据需要决定是否重命名
3. 📊 **监控新文件** - 注意新下载的PDF是否还有日期错误

### 长期建议

1. **检查上游流程**：
   - 找出谁在给PDF命名
   - 修复命名脚本的日期逻辑

2. **添加验证**：
   - 在下载/命名时检查日期合理性
   - 自动标记异常日期的文件

3. **文档化命名规范**：
   - 明确PDF文件命名格式
   - 日期应该从何处获取

---

**生成日期：** 2025-10-27
**总计错误文件：** 10个PDF
**已修复输出：** 14个项目（8个JSON + 6个文件夹）
**源文件状态：** 未修改（保持原样）
