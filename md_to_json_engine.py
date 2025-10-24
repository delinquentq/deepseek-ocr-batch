#!/usr/bin/env python3
"""
Markdown到JSON的规则引擎
使用规则处理结构化数据，避免大模型调用，大幅提升速度和降低成本

核心功能：
1. 解析Markdown文档结构
2. 提取元数据（标题、日期、作者等）
3. 分割段落和章节
4. 识别表格
5. 提取数值数据
6. 生成符合Schema v1.3.1的JSON
"""

import re
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MarkdownParser:
    """Markdown文档解析器"""

    def __init__(self):
        self.page_separator = r'\n\n--- Page (\d+) ---\n\n'

    def split_by_pages(self, markdown: str) -> List[Tuple[int, str]]:
        """按页分割Markdown"""
        pages = []
        parts = re.split(self.page_separator, markdown)

        # 第一部分是第0页（封面）
        if parts[0].strip():
            pages.append((0, parts[0].strip()))

        # 后续部分：奇数索引是页码，偶数索引是内容
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                page_num = int(parts[i])
                content = parts[i + 1].strip()
                pages.append((page_num, content))

        return pages

    def extract_headings(self, text: str) -> List[Dict]:
        """提取标题"""
        headings = []
        # 匹配 # 标题
        pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(pattern, text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append({
                "level": level,
                "title": title
            })
        return headings

    def extract_paragraphs(self, text: str, page_num: int) -> List[Dict]:
        """提取段落"""
        paragraphs = []

        # 移除图片引用
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除特殊标记
        text = re.sub(r'<｜end▁of▁sentence｜>', '', text)

        # 按空行分割段落
        parts = text.split('\n\n')

        for idx, part in enumerate(parts):
            part = part.strip()
            # 过滤太短的段落和标题
            if len(part) < 20 or part.startswith('#'):
                continue

            # 生成passage_id
            passage_id = hashlib.md5(f"{page_num}_{idx}_{part[:50]}".encode()).hexdigest()[:16]

            paragraphs.append({
                "passage_id": passage_id,
                "text": part,
                "page": page_num,
                "section": None  # 可以后续关联章节
            })

        return paragraphs

    def extract_tables_from_markdown(self, text: str, page_num: int) -> List[Dict]:
        """从Markdown中提取表格"""
        tables = []

        # Markdown表格格式：| col1 | col2 |
        table_pattern = r'(\|.+\|[\r\n]+\|[-:\s|]+\|[\r\n]+(?:\|.+\|[\r\n]+)+)'

        for idx, match in enumerate(re.finditer(table_pattern, text)):
            table_text = match.group(1)
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]

            if len(lines) < 3:  # 至少需要表头、分隔符、一行数据
                continue

            # 解析表头
            header_line = lines[0]
            columns = [col.strip() for col in header_line.split('|')[1:-1]]

            # 解析数据行（跳过分隔符行）
            rows = []
            for line in lines[2:]:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(cells) == len(columns):
                    row_dict = {columns[i]: cells[i] for i in range(len(columns))}
                    rows.append(row_dict)

            if rows:
                table_id = hashlib.md5(f"{page_num}_{idx}_{table_text[:50]}".encode()).hexdigest()[:16]
                tables.append({
                    "table_id": table_id,
                    "title": f"Table on page {page_num}",
                    "page": page_num,
                    "columns": columns,
                    "rows": rows,
                    "provenance": {"page": page_num}
                })

        return tables


class MetadataExtractor:
    """元数据提取器"""

    def extract_title(self, markdown: str) -> str:
        """提取文档标题"""
        # 尝试提取第一个一级标题
        match = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # 尝试提取前几行的粗体文本
        lines = markdown.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # 移除Markdown格式
                clean_line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
                clean_line = re.sub(r'<[^>]+>', '', clean_line)
                if len(clean_line) > 10:
                    return clean_line[:200]

        return "Untitled Document"

    def extract_date(self, markdown: str) -> Optional[str]:
        """提取日期"""
        # 常见日期格式
        patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',  # 2025-09-03
            r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',  # 3 September 2025
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',  # 09/03/2025
        ]

        for pattern in patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def extract_companies(self, markdown: str) -> List[str]:
        """提取公司名称"""
        companies = set()

        # 常见公司后缀
        company_patterns = [
            r'\b([A-Z][A-Za-z\s&]+(?:Inc|Corp|Corporation|Ltd|Limited|Group|Company|Co)\.?)\b',
            r'\b([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+)\b',  # 两个大写开头的词
        ]

        for pattern in company_patterns:
            matches = re.findall(pattern, markdown)
            for match in matches:
                company = match.strip()
                # 过滤太短或太长的
                if 3 < len(company) < 50 and not company.isupper():
                    companies.add(company)

        return sorted(list(companies))[:20]  # 最多返回20个

    def extract_authors(self, markdown: str) -> List[str]:
        """提取作者"""
        authors = []

        # 查找邮箱地址附近的名字
        email_pattern = r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+[+\d\s()-]+\s+[\w.]+@[\w.]+'
        matches = re.findall(email_pattern, markdown)
        authors.extend(matches)

        return list(set(authors))[:10]  # 去重，最多10个


class NumericalDataExtractor:
    """数值数据提取器"""

    def extract_numerical_data(self, markdown: str, page_num: int) -> List[Dict]:
        """提取数值数据"""
        numerical_data = []

        # 提取百分比
        percentage_pattern = r'([+-]?\d+\.?\d*)\s*%'
        for match in re.finditer(percentage_pattern, markdown):
            value = match.group(1)
            context = self._get_context(markdown, match.start(), match.end())

            num_id = hashlib.md5(f"{page_num}_{value}_{context[:30]}".encode()).hexdigest()[:16]
            numerical_data.append({
                "num_id": num_id,
                "value": float(value) / 100,  # 转换为0-1
                "unit": "percentage",
                "metric_type": "percentage",
                "context": context,
                "provenance": {"page": page_num}
            })

        # 提取货币金额
        currency_pattern = r'([$€£¥])\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*([BMK])?'
        for match in re.finditer(currency_pattern, markdown):
            symbol = match.group(1)
            value = match.group(2).replace(',', '')
            multiplier = match.group(3)
            context = self._get_context(markdown, match.start(), match.end())

            # 转换倍数
            if multiplier == 'B':
                value = float(value) * 1e9
            elif multiplier == 'M':
                value = float(value) * 1e6
            elif multiplier == 'K':
                value = float(value) * 1e3
            else:
                value = float(value)

            num_id = hashlib.md5(f"{page_num}_{value}_{context[:30]}".encode()).hexdigest()[:16]
            numerical_data.append({
                "num_id": num_id,
                "value": value,
                "unit": symbol,
                "metric_type": "currency",
                "context": context,
                "provenance": {"page": page_num}
            })

        return numerical_data[:50]  # 限制数量

    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """获取数值周围的上下文"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end].strip()
        # 清理换行和多余空格
        context = re.sub(r'\s+', ' ', context)
        return context[:200]


class EntityExtractor:
    """实体提取器"""

    def extract_entities(self, markdown: str, companies: List[str]) -> List[Dict]:
        """提取实体"""
        entities = []

        # 从公司列表创建实体
        for idx, company in enumerate(companies):
            entity_id = hashlib.md5(company.encode()).hexdigest()[:16]
            entities.append({
                "entity_id": entity_id,
                "name": company,
                "type": "company",
                "aliases": []
            })

        return entities


class MarkdownToJsonEngine:
    """Markdown到JSON的主引擎"""

    def __init__(self):
        self.parser = MarkdownParser()
        self.metadata_extractor = MetadataExtractor()
        self.numerical_extractor = NumericalDataExtractor()
        self.entity_extractor = EntityExtractor()

    def convert(self, markdown_content: str, pdf_name: str,
                date_str: Optional[str] = None,
                publication: Optional[str] = None) -> Dict:
        """
        将Markdown转换为JSON Schema v1.3.1格式

        Args:
            markdown_content: Markdown文本
            pdf_name: PDF文件名
            date_str: 日期字符串
            publication: 出版物名称

        Returns:
            符合Schema v1.3.1的JSON字典
        """
        logger.info(f"开始转换Markdown到JSON: {pdf_name}")

        # 1. 提取元数据
        title = self.metadata_extractor.extract_title(markdown_content)
        extracted_date = self.metadata_extractor.extract_date(markdown_content)
        companies = self.metadata_extractor.extract_companies(markdown_content)
        authors = self.metadata_extractor.extract_authors(markdown_content)

        # 2. 按页分割
        pages = self.parser.split_by_pages(markdown_content)
        page_count = len(pages)

        # 3. 提取所有段落
        all_passages = []
        for page_num, page_content in pages:
            passages = self.parser.extract_paragraphs(page_content, page_num)
            all_passages.extend(passages)

        # 4. 提取所有表格
        all_tables = []
        for page_num, page_content in pages:
            tables = self.parser.extract_tables_from_markdown(page_content, page_num)
            all_tables.extend(tables)

        # 5. 提取数值数据
        all_numerical_data = []
        for page_num, page_content in pages:
            numerical_data = self.numerical_extractor.extract_numerical_data(page_content, page_num)
            all_numerical_data.extend(numerical_data)

        # 6. 提取实体
        entities = self.entity_extractor.extract_entities(markdown_content, companies)

        # 7. 构建JSON结构
        doc_id = hashlib.md5(pdf_name.encode()).hexdigest()

        result = {
            "schema_version": "1.3.1",
            "doc": {
                "doc_id": doc_id,
                "title": title,
                "source_uri": f"{publication}/{pdf_name}" if publication else pdf_name,
                "language": "en",
                "timestamps": {
                    "ingested_at": datetime.now().isoformat(),
                    "extracted_at": datetime.now().isoformat()
                },
                "extraction_run": {
                    "vision_model": "deepseek-ai/DeepSeek-OCR",
                    "synthesis_model": "rule-based-engine",
                    "pipeline_steps": ["ocr", "rule_extraction", "figure_vision"],
                    "processing_metadata": {
                        "pages_processed": page_count,
                        "successful_pages": page_count,
                        "date": date_str or extracted_date,
                        "publication": publication,
                        "authors": authors
                    }
                }
            },
            "passages": all_passages,
            "entities": entities,
            "data": {
                "figures": [],  # 将由视觉模型填充
                "tables": all_tables,
                "numerical_data": all_numerical_data,
                "claims": [],  # 可选：后续添加
                "relations": [],  # 可选：后续添加
                "extraction_summary": {
                    "figures_count": 0,  # 将更新
                    "tables_count": len(all_tables),
                    "numerical_data_count": len(all_numerical_data),
                    "passages_count": len(all_passages),
                    "entities_count": len(entities)
                }
            }
        }

        logger.info(f"✓ Markdown转换完成: {len(all_passages)}段落, {len(all_tables)}表格, {len(all_numerical_data)}数值")
        return result


# 测试代码
if __name__ == "__main__":
    # 简单测试
    test_md = """
# Test Document

This is a test document with **Apple Inc.** reporting revenue of $89.5B, up 6% YoY.

## Financial Results

| Metric | 2023 | 2024 |
|--------|------|------|
| Revenue | $100M | $120M |
| Profit | 15% | 18% |

--- Page 1 ---

More content here.
"""

    engine = MarkdownToJsonEngine()
    result = engine.convert(test_md, "test.pdf")

    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
