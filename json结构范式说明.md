让我为您详细说明这个项目的JSON输出结构范式。

[3 tools called]

# 📋 项目JSON输出结构范式详细说明

## 一、顶层结构概览

这个项目的JSON输出文件包含**5个顶层字段**，每个字段都有严格的类型和嵌套结构：

```json
{
  "_id": "string - 文件唯一标识符",
  "source": { /* 来源信息 */ },
  "report": { /* 报告元数据和内容 */ },
  "data": { /* 提取的数据 */ },
  "query_capabilities": { /* 查询能力说明 */ }
}
```

---

## 二、详细字段说明

### 1. `_id` 字段
**类型**: `string`  
**说明**: PDF文件的唯一哈希标识符，用于防止重复处理  
**示例**: `"a8f3e9d2c1b4567890abcdef12345678"`

---

### 2. `source` 字段 - 来源信息

**结构**:
```json
{
  "file_name": "string - 原始PDF文件名",
  "processing_metadata": {
    "vision_model": "string - 视觉模型名称",
    "synthesis_model": "string - 合成模型名称", 
    "validation_model": "string - 验证模型名称",
    "processed_at": "string - ISO 8601时间戳",
    "pages_processed": "int - 处理的总页数",
    "successful_pages": "int - 成功处理的页数"
  }
}
```

**实例**:
```json
{
  "file_name": "Apple_Q4_2024_Analysis.pdf",
  "processing_metadata": {
    "vision_model": "gemini/gemini-2.5-flash",
    "synthesis_model": "google/gemini-2.5-flash",
    "validation_model": "google/gemini-2.5-flash",
    "processed_at": "2024-10-21T15:30:45Z",
    "pages_processed": 25,
    "successful_pages": 24
  }
}
```

---

### 3. `report` 字段 - 报告元数据

**结构**:
```json
{
  "title": "string - 报告标题",
  "report_date": "string|null - 报告日期 YYYY-MM-DD格式",
  "report_type": "string - 报告类型",
  "symbols": ["array - 股票代码列表"],
  "sector": "string|null - 行业/板块",
  "content": "string - 完整合成报告内容",
  "word_count": "int - 字数统计"
}
```

**字段详解**:

- **`report_type`** 必须是以下之一：
  - `"company"` - 公司研究报告
  - `"sector"` - 行业/板块报告
  - `"macro"` - 宏观经济报告
  - `"strategy"` - 策略报告

- **`symbols`**: 最多1-2个主要股票代码，使用标准ticker格式（如 `["AAPL", "MSFT"]`）

- **`sector`**: 具体行业名称（如 `"Technology"`, `"Healthcare"`, `"Financial Services"`）

**实例**:
```json
{
  "title": "Apple Inc. Q4 2024 Financial Analysis",
  "report_date": "2024-10-15",
  "report_type": "company",
  "symbols": ["AAPL"],
  "sector": "Technology",
  "content": "Executive Summary\n\nApple Inc. delivered strong Q4 2024 results...",
  "word_count": 3542
}
```

---

### 4. `data` 字段 - 提取的数据（核心部分）

**结构**:
```json
{
  "figures": [ /* 图表数组 */ ],
  "numerical_data": [ /* 数值数据数组 */ ],
  "companies": [ /* 提及的公司列表 */ ],
  "key_metrics": [ /* 关键指标列表 */ ],
  "extraction_summary": { /* 提取摘要 */ }
}
```

#### 4.1 `figures` 数组 - 图表数据

每个图表对象的结构：

```json
{
  "figure_id": "string - 描述性ID（snake_case）",
  "type": "string - 图表类型",
  "title": "string - 图表标题",
  "description": "string - 图表描述",
  "data": {
    // 根据图表类型不同，结构不同
  },
  "source_page": "int - 来源页码"
}
```

**图表类型 (`type`)** 包括：
- `"bar_chart"` - 柱状图
- `"line_chart"` - 折线图
- `"pie_chart"` - 饼图
- `"table"` - 表格
- `"scatter_chart"` - 散点图
- `"area_chart"` - 面积图

**`data` 字段结构**根据图表类型不同：

##### A. 柱状图/折线图/饼图 - 使用 labels + series 结构

```json
{
  "data": {
    "labels": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
    "series": [
      {
        "name": "Revenue",
        "values": [100, 120, 135, 150],
        "unit": "M"
      },
      {
        "name": "Profit",
        "values": [10, 15, 18, 22],
        "unit": "M"
      }
    ]
  }
}
```

**字段说明**:
- `labels`: X轴标签数组
- `series`: 数据系列数组，每个系列包含：
  - `name`: 系列名称
  - `values`: 数值数组（必须与labels长度一致）
  - `unit`: 单位（如 `"M"`, `"%"`, `"units"`）

##### B. 表格 - 使用 rows 结构

```json
{
  "data": {
    "rows": [
      {
        "metric": "Revenue",
        "2023": "500",
        "2024": "650",
        "2025E": "800"
      },
      {
        "metric": "EBITDA",
        "2023": "100",
        "2024": "130",
        "2025E": "160"
      },
      {
        "metric": "Margin",
        "2023": "20%",
        "2024": "20%",
        "2025E": "20%"
      }
    ]
  }
}
```

**字段说明**:
- `rows`: 表格行数组，每行是一个对象
- 每个行对象的键（列名）可以是动态的
- 所有行应具有相同的键集合

##### 完整图表示例

```json
{
  "figure_id": "quarterly_revenue_growth",
  "type": "bar_chart",
  "title": "Quarterly Revenue Growth 2024",
  "description": "Year-over-year revenue growth by quarter",
  "data": {
    "labels": ["Q1", "Q2", "Q3", "Q4"],
    "series": [
      {
        "name": "Revenue Growth",
        "values": [12.5, 15.3, 14.8, 18.2],
        "unit": "%"
      }
    ]
  },
  "source_page": 8
}
```

#### 4.2 `numerical_data` 数组 - 数值数据点

每个数值数据对象的结构：

```json
{
  "metric": "string - 指标名",
  "value": "number|string - 数值（可保留原格式）",
  "unit": "string|null - 单位",
  "time_period": "string|null - 时间段"
}
```

```

<File before editing>
```markdown
让我为您详细说明这个项目的JSON输出结构范式。

[3 tools called]

# 📋 项目JSON输出结构范式详细说明

## 一、顶层结构概览

这个项目的JSON输出文件包含**5个顶层字段**，每个字段都有严格的类型和嵌套结构：

```json
{
  "_id": "string - 文件唯一标识符",
  "source": { /* 来源信息 */ },
  "report": { /* 报告元数据和内容 */ },
  "data": { /* 提取的数据 */ },
  "query_capabilities": { /* 查询能力说明 */ }
}
```

---

## 二、详细字段说明

### 1. `_id` 字段
**类型**: `string`  
**说明**: PDF文件的唯一哈希标识符，用于防止重复处理  
**示例**: `"a8f3e9d2c1b4567890abcdef12345678"`

---

### 2. `source` 字段 - 来源信息

**结构**:
```json
{
  "file_name": "string - 原始PDF文件名",
  "processing_metadata": {
    "vision_model": "string - 视觉模型名称",
    "synthesis_model": "string - 合成模型名称", 
    "validation_model": "string - 验证模型名称",
    "processed_at": "string - ISO 8601时间戳",
    "pages_processed": "int - 处理的总页数",
    "successful_pages": "int - 成功处理的页数"
  }
}
```

**实例**:
```json
{
  "file_name": "Apple_Q4_2024_Analysis.pdf",
  "processing_metadata": {
    "vision_model": "qwen/qwen-2.5-vl-72b-instruct",
    "synthesis_model": "google/gemini-2.5-flash",
    "validation_model": "google/gemini-2.0-flash-exp:free",
    "processed_at": "2024-10-21T15:30:45Z",
    "pages_processed": 25,
    "successful_pages": 24
  }
}
```

---

### 3. `report` 字段 - 报告元数据

**结构**:
```json
{
  "title": "string - 报告标题",
  "report_date": "string|null - 报告日期 YYYY-MM-DD格式",
  "report_type": "string - 报告类型",
  "symbols": ["array - 股票代码列表"],
  "sector": "string|null - 行业/板块",
  "content": "string - 完整合成报告内容",
  "word_count": "int - 字数统计"
}
```

**字段详解**:

- **`report_type`** 必须是以下之一：
  - `"company"` - 公司研究报告
  - `"sector"` - 行业/板块报告
  - `"macro"` - 宏观经济报告
  - `"strategy"` - 策略报告

- **`symbols`**: 最多1-2个主要股票代码，使用标准ticker格式（如 `["AAPL", "MSFT"]`）

- **`sector`**: 具体行业名称（如 `"Technology"`, `"Healthcare"`, `"Financial Services"`）

**实例**:
```json
{
  "title": "Apple Inc. Q4 2024 Financial Analysis",
  "report_date": "2024-10-15",
  "report_type": "company",
  "symbols": ["AAPL"],
  "sector": "Technology",
  "content": "Executive Summary\n\nApple Inc. delivered strong Q4 2024 results...",
  "word_count": 3542
}
```

---

### 4. `data` 字段 - 提取的数据（核心部分）

**结构**:
```json
{
  "figures": [ /* 图表数组 */ ],
  "numerical_data": [ /* 数值数据数组 */ ],
  "companies": [ /* 提及的公司列表 */ ],
  "key_metrics": [ /* 关键指标列表 */ ],
  "extraction_summary": { /* 提取摘要 */ }
}
```

#### 4.1 `figures` 数组 - 图表数据

每个图表对象的结构：

```json
{
  "figure_id": "string - 描述性ID（snake_case）",
  "type": "string - 图表类型",
  "title": "string - 图表标题",
  "description": "string - 图表描述",
  "data": {
    // 根据图表类型不同，结构不同
  },
  "source_page": "int - 来源页码"
}
```

**图表类型 (`type`)** 包括：
- `"bar_chart"` - 柱状图
- `"line_chart"` - 折线图
- `"pie_chart"` - 饼图
- `"table"` - 表格
- `"scatter_chart"` - 散点图
- `"area_chart"` - 面积图

**`data` 字段结构**根据图表类型不同：

##### A. 柱状图/折线图/饼图 - 使用 labels + series 结构

```json
{
  "data": {
    "labels": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
    "series": [
      {
        "name": "Revenue",
        "values": [100, 120, 135, 150],
        "unit": "$M"
      },
      {
        "name": "Profit",
        "values": [10, 15, 18, 22],
        "unit": "$M"
      }
    ]
  }
}
```

**字段说明**:
- `labels`: X轴标签数组
- `series`: 数据系列数组，每个系列包含：
  - `name`: 系列名称
  - `values`: 数值数组（必须与labels长度一致）
  - `unit`: 单位（如 `"$M"`, `"%"`, `"units"`）

##### B. 表格 - 使用 rows 结构

```json
{
  "data": {
    "rows": [
      {
        "metric": "Revenue",
        "2023": "$500M",
        "2024": "$650M",
        "2025E": "$800M"
      },
      {
        "metric": "EBITDA",
        "2023": "$100M",
        "2024": "$130M",
        "2025E": "$160M"
      },
      {
        "metric": "Margin",
        "2023": "20%",
        "2024": "20%",
        "2025E": "20%"
      }
    ]
  }
}
```

**字段说明**:
- `rows`: 表格行数组，每行是一个对象
- 每个行对象的键（列名）可以是动态的
- 所有行应具有相同的键集合

##### 完整图表示例

```json
{
  "figure_id": "quarterly_revenue_growth",
  "type": "bar_chart",
  "title": "Quarterly Revenue Growth 2024",
  "description": "Year-over-year revenue growth by quarter",
  "data": {
    "labels": ["Q1", "Q2", "Q3", "Q4"],
    "series": [
      {
        "name": "Revenue Growth",
        "values": [12.5, 15.3, 14.8, 18.2],
        "unit": "%"
      }
    ]
  },
  "source_page": 8
}
```

#### 4.2 `numerical_data` 数组 - 数值数据点

每个数值数据对象的结构：

```json
{
  "value": "string - 数值（保持原格式）",
  "figure_id": "string|null - 关联的图表ID",
  "context": "string - 数据上下文/说明",
  "metric_type": "string - 指标类型",
  "source_page": "int - 来源页码"
}
```

**`metric_type`** 包括：
- `"percentage"` - 百分比
- `"currency"` - 货币
- `"ratio"` - 比率
- `"count"` - 计数
- `"decimal"` - 小数

**示例**:
```json
[
  {
    "value": "15.2%",
    "figure_id": "quarterly_revenue_growth",
    "context": "Q4 revenue growth rate",
    "metric_type": "percentage",
    "source_page": 8
  },
  {
    "value": "$2.5B",
    "figure_id": null,
    "context": "Total annual revenue",
    "metric_type": "currency",
    "source_page": 3
  }
]
```

#### 4.3 `companies` 数组

```json
["Apple Inc", "Microsoft Corp", "Amazon.com Inc"]
```

提及的公司名称列表（字符串数组）

#### 4.4 `key_metrics` 数组

```json
["revenue", "profit_margin", "eps", "market_share", "growth_rate"]
```

文档中的关键财务指标（字符串数组）

#### 4.5 `extraction_summary` 对象

```json
{
  "figures_count": 12,
  "numerical_data_count": 38,
  "companies_mentioned": 5,
  "figures_with_linked_data": 10,
  "validation_summary": {
    "original_figures": 15,
    "kept_figures": 12,
    "original_numerical": 45,
    "kept_numerical": 38,
    "validation_method": "gemini_25_flash_intelligent_filtering",
    "data_accuracy_rate": 84.4
  }
}
```

**字段说明**:
- `figures_count`: 提取的图表总数
- `numerical_data_count`: 提取的数值数据点总数
- `companies_mentioned`: 提及的公司数量
- `figures_with_linked_data`: 带有完整data字段的图表数
- `validation_summary`: 数据验证摘要
  - `original_figures/numerical`: 原始提取数量
  - `kept_figures/numerical`: 验证后保留数量
  - `validation_method`: 使用的验证方法
  - `data_accuracy_rate`: 数据准确率

---

### 5. `query_capabilities` 字段

```json
{
  "description": "This document supports figure-to-data linking queries",
  "searchable_fields": [
    "report.title",
    "report.content",
    "data.figures.title",
    "data.figures.description",
    "data.numerical_data.context"
  ],
  "figure_data_available": true,
  "can_recreate_charts": true
}
```

说明文档的查询能力和可用功能。

---

## 三、完整JSON示例

```json
{
  "_id": "a8f3e9d2c1b4567890abcdef12345678",
  "source": {
    "file_name": "Apple_Q4_2024_Analysis.pdf",
    "processing_metadata": {
      "vision_model": "qwen/qwen-2.5-vl-72b-instruct",
      "synthesis_model": "google/gemini-2.5-flash",
      "validation_model": "google/gemini-2.0-flash-exp:free",
      "processed_at": "2024-10-21T15:30:45Z",
      "pages_processed": 25,
      "successful_pages": 24
    }
  },
  "report": {
    "title": "Apple Inc. Q4 2024 Financial Analysis",
    "report_date": "2024-10-15",
    "report_type": "company",
    "symbols": ["AAPL"],
    "sector": "Technology",
    "content": "Executive Summary\n\nApple Inc. delivered strong Q4 2024 results with revenue reaching $89.5B, up 6% YoY...\n\n[完整报告内容]",
    "word_count": 3542
  },
  "data": {
    "figures": [
      {
        "figure_id": "quarterly_revenue_growth",
        "type": "bar_chart",
        "title": "Quarterly Revenue Growth 2024",
        "description": "Year-over-year revenue growth by quarter",
        "data": {
          "labels": ["Q1", "Q2", "Q3", "Q4"],
          "series": [
            {
              "name": "Revenue Growth",
              "values": [12.5, 15.3, 14.8, 18.2],
              "unit": "%"
            }
          ]
        },
        "source_page": 8
      },
      {
        "figure_id": "product_revenue_breakdown",
        "type": "pie_chart",
        "title": "Revenue by Product Category",
        "description": "Q4 2024 revenue distribution",
        "data": {
          "labels": ["iPhone", "Services", "Mac", "iPad", "Wearables"],
          "series": [
            {
              "name": "Revenue Share",
              "values": [52, 22, 11, 7, 8],
              "unit": "%"
            }
          ]
        },
        "source_page": 10
      },
      {
        "figure_id": "financial_metrics_table",
        "type": "table",
        "title": "Key Financial Metrics",
        "description": "Historical and projected financial metrics",
        "data": {
          "rows": [
            {"metric": "Revenue", "2023": "$383B", "2024": "$405B", "2025E": "$430B"},
            {"metric": "Net Income", "2023": "$97B", "2024": "$105B", "2025E": "$112B"},
            {"metric": "EPS", "2023": "$6.16", "2024": "$6.70", "2025E": "$7.20"},
            {"metric": "Gross Margin", "2023": "44.1%", "2024": "45.2%", "2025E": "45.5%"}
          ]
        },
        "source_page": 15
      }
    ],
    "numerical_data": [
      {
        "value": "18.2%",
        "figure_id": "quarterly_revenue_growth",
        "context": "Q4 revenue growth rate",
        "metric_type": "percentage",
        "source_page": 8
      },
      {
        "value": "$89.5B",
        "figure_id": null,
        "context": "Q4 2024 total revenue",
        "metric_type": "currency",
        "source_page": 3
      },
      {
        "value": "52%",
        "figure_id": "product_revenue_breakdown",
        "context": "iPhone revenue contribution",
        "metric_type": "percentage",
        "source_page": 10
      },
      {
        "value": "$6.70",
        "figure_id": "financial_metrics_table",
        "context": "2024 earnings per share",
        "metric_type": "currency",
        "source_page": 15
      }
    ],
    "companies": [
      "Apple Inc",
      "Samsung Electronics",
      "Qualcomm"
    ],
    "key_metrics": [
      "revenue",
      "net_income",
      "eps",
      "gross_margin",
      "market_share"
    ],
    "extraction_summary": {
      "figures_count": 12,
      "numerical_data_count": 38,
      "companies_mentioned": 5,
      "figures_with_linked_data": 10,
      "validation_summary": {
        "original_figures": 15,
        "kept_figures": 12,
        "original_numerical": 45,
        "kept_numerical": 38,
        "validation_method": "gemini_25_flash_intelligent_filtering",
        "data_accuracy_rate": 84.4
      }
    }
  },
  "query_capabilities": {
    "description": "This document supports figure-to-data linking queries",
    "searchable_fields": [
      "report.title",
      "report.content",
      "data.figures.title",
      "data.figures.description"
    ],
    "figure_data_available": true,
    "can_recreate_charts": true
  }
}
```

---

## 四、关键设计原则

### 1. **图表数据可重建性**
- 所有图表的 `data` 字段包含完整的原始数据
- 可以从JSON直接重建图表，无需回查PDF
- `labels` 和 `values` 数组长度必须匹配

### 2. **数据关联性**
- 通过 `figure_id` 建立数值数据与图表的关联
- `numerical_data` 中的每个数据点都标明来源

### 3. **类型严格性**
- 每个字段都有明确的类型定义
- 使用schema验证确保数据质量
- 缺失字段使用合理的默认值（如空数组、null）

### 4. **可查询性**
- 结构化设计支持复杂的数据库查询
- 可按图表类型、指标类型、公司等多维度检索
- 支持时间序列分析和跨文档比较

---

## 五、代码中的Schema定义位置

在 `ultra_fast_processor.py` 文件中：

- **第738-772行**: Extraction Prompt（定义提取时的格式）
- **第1422-1511行**: `_validate_output_document_schema()` 方法（运行时验证）
- **第921-981行**: `_extract_nosql_metadata_with_llm()` 方法（元数据提取）

这个schema设计既保证了数据的完整性和可用性，又便于数据库存储和查询操作。