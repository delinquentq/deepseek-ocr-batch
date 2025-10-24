#!/usr/bin/env python3
"""
Markdown文档清洗引擎
专门用于清理金融报告中的无效内容，只保留正文分析部分

清洗规则：
1. 删除合规声明页（Analyst Certification, Important Disclosures等）
2. 删除法律页（各地区法律声明）
3. 删除商标与版权页
4. 删除分析师联系信息
5. 删除会议日程信息
6. 删除Disclaimer和风险披露
7. 保留核心分析内容（Executive Summary, Analysis, Outlook等）
"""

import re
import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CleaningRule:
    """清洗规则"""
    name: str
    pattern: str  # 正则表达式
    description: str
    is_section_header: bool = False  # 是否是章节标题（需要删除整个章节）


class MarkdownCleaner:
    """Markdown文档清洗器"""

    def __init__(self):
        self.rules = self._init_cleaning_rules()

    def _init_cleaning_rules(self) -> List[CleaningRule]:
        """初始化清洗规则"""
        rules = [
            # 1. 合规声明章节（章节标题，删除整个章节）
            CleaningRule(
                name="analyst_certification",
                pattern=r"^#{1,3}\s*Analyst['\s]*(?:s)?\s*Certification[s]?",
                description="分析师认证声明",
                is_section_header=True
            ),
            CleaningRule(
                name="important_disclosures",
                pattern=r"^#{1,3}\s*Important\s+Disclosure[s]?",
                description="重要披露声明",
                is_section_header=True
            ),
            CleaningRule(
                name="risk_disclosure",
                pattern=r"^#{1,3}\s*Risk\s+Disclosure[s]?",
                description="风险披露",
                is_section_header=True
            ),
            CleaningRule(
                name="availability_disclosures",
                pattern=r"^#{1,3}\s*Availability\s+of\s+Disclosure[s]?",
                description="披露可用性",
                is_section_header=True
            ),
            CleaningRule(
                name="information_sources",
                pattern=r"^#{1,3}\s*(?:Disclosure[s]?\s+regarding\s+)?Information\s+Source[s]?",
                description="信息来源披露",
                is_section_header=True
            ),
            CleaningRule(
                name="rating_system",
                pattern=r"^#{1,3}\s*(?:Guide\s+to\s+the\s+)?.*Rating\s+System",
                description="评级系统说明",
                is_section_header=True
            ),
            CleaningRule(
                name="price_target",
                pattern=r"^#{1,3}\s*(?:Guide\s+to\s+the\s+)?.*Price\s+Target",
                description="目标价说明",
                is_section_header=True
            ),
            CleaningRule(
                name="distribution_ratings",
                pattern=r"^#{1,3}\s*Distribution\s+of\s+Rating[s]?",
                description="评级分布",
                is_section_header=True
            ),
            CleaningRule(
                name="disclosure_legend",
                pattern=r"^#{1,3}\s*Disclosure\s+Legend",
                description="披露图例",
                is_section_header=True
            ),
            CleaningRule(
                name="legal_entities",
                pattern=r"^#{1,3}\s*Legal\s+(?:entities|Entities)\s+(?:involved|Involved)\s+in\s+(?:producing|Producing)",
                description="法律实体声明",
                is_section_header=True
            ),

            # 2. 地区法律页（章节标题）
            CleaningRule(
                name="regional_legal_uk",
                pattern=r"^#{1,3}\s*(?:United\s+Kingdom|UK)\s*(?:/\s*EEA)?",
                description="英国/欧洲法律页",
                is_section_header=True
            ),
            CleaningRule(
                name="regional_legal_americas",
                pattern=r"^#{1,3}\s*Americas?",
                description="美洲法律页",
                is_section_header=True
            ),
            CleaningRule(
                name="regional_legal_japan",
                pattern=r"^#{1,3}\s*Japan",
                description="日本法律页",
                is_section_header=True
            ),
            CleaningRule(
                name="regional_legal_hongkong",
                pattern=r"^#{1,3}\s*Hong\s+Kong",
                description="香港法律页",
                is_section_header=True
            ),
            CleaningRule(
                name="regional_legal_middleeast",
                pattern=r"^#{1,3}\s*Middle\s+East",
                description="中东法律页",
                is_section_header=True
            ),
            CleaningRule(
                name="regional_legal_australia",
                pattern=r"^#{1,3}\s*Australia",
                description="澳大利亚法律页",
                is_section_header=True
            ),
            CleaningRule(
                name="regional_legal_singapore",
                pattern=r"^#{1,3}\s*Singapore",
                description="新加坡法律页",
                is_section_header=True
            ),

            # 3. 商标与版权（段落级别，删除匹配段落）
            CleaningRule(
                name="trademark_bloomberg",
                pattern=r"Bloomberg®\s+is\s+a\s+trademark",
                description="Bloomberg商标声明",
                is_section_header=False
            ),
            CleaningRule(
                name="copyright_barclays",
                pattern=r"(?:Barclays\s+Bank\s+PLC|Barclays\s+Capital\s+Inc\.|©\s+Copyright\s+Barclays)",
                description="Barclays版权声明",
                is_section_header=False
            ),
            CleaningRule(
                name="copyright_generic",
                pattern=r"©\s+Copyright\s+\d{4}",
                description="通用版权声明",
                is_section_header=False
            ),

            # 4. 分析师联系信息（段落级别）
            CleaningRule(
                name="analyst_contact",
                pattern=r"(?:Tel|Phone|Email):\s*[+\d\s()-]+|[\w.]+@[\w.]+",
                description="分析师联系方式",
                is_section_header=False
            ),

            # 5. 会议日程（章节标题）
            CleaningRule(
                name="conference_schedule",
                pattern=r"^#{1,3}\s*(?:Conference\s+)?(?:Schedule|Calendar|Agenda)",
                description="会议日程",
                is_section_header=True
            ),

            # 6. Disclaimer（章节标题）
            CleaningRule(
                name="disclaimer",
                pattern=r"^#{1,3}\s*Disclaimer[s]?",
                description="免责声明",
                is_section_header=True
            ),
        ]

        return rules

    def clean(self, markdown_content: str) -> Tuple[str, Dict]:
        """
        清洗Markdown文档

        Args:
            markdown_content: 原始Markdown文本

        Returns:
            (cleaned_content, statistics)
        """
        logger.info("开始清洗Markdown文档...")

        original_length = len(markdown_content)
        stats = {
            "original_length": original_length,
            "removed_sections": [],
            "removed_paragraphs": 0,
            "final_length": 0
        }

        # 按页分割
        pages = self._split_by_pages(markdown_content)
        cleaned_pages = []

        for page_num, page_content in pages:
            cleaned_content = self._clean_page(page_content, stats)
            if cleaned_content.strip():  # 只保留非空页
                cleaned_pages.append((page_num, cleaned_content))

        # 重新组合
        result = self._merge_pages(cleaned_pages)

        stats["final_length"] = len(result)
        stats["reduction_ratio"] = 1 - (stats["final_length"] / original_length) if original_length > 0 else 0

        logger.info(f"✓ 清洗完成: 原始{original_length}字符 → 清洗后{stats['final_length']}字符 (减少{stats['reduction_ratio']*100:.1f}%)")
        logger.info(f"  删除章节: {len(stats['removed_sections'])}个")
        logger.info(f"  删除段落: {stats['removed_paragraphs']}个")

        return result, stats

    def _split_by_pages(self, markdown: str) -> List[Tuple[int, str]]:
        """按页分割Markdown"""
        page_separator = r'\n\n--- Page (\d+) ---\n\n'
        pages = []
        parts = re.split(page_separator, markdown)

        # 第一部分是第0页
        if parts[0].strip():
            pages.append((0, parts[0].strip()))

        # 后续部分：奇数索引是页码，偶数索引是内容
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                page_num = int(parts[i])
                content = parts[i + 1].strip()
                pages.append((page_num, content))

        return pages

    def _clean_page(self, page_content: str, stats: Dict) -> str:
        """清洗单页内容"""
        # 1. 删除章节级别的无效内容
        cleaned = self._remove_invalid_sections(page_content, stats)

        # 2. 删除段落级别的无效内容
        cleaned = self._remove_invalid_paragraphs(cleaned, stats)

        return cleaned

    def _remove_invalid_sections(self, content: str, stats: Dict) -> str:
        """删除无效章节"""
        lines = content.split('\n')
        result_lines = []
        skip_until_next_section = False
        current_section_name = None

        for i, line in enumerate(lines):
            # 检查是否是章节标题
            is_header = line.strip().startswith('#')

            if is_header:
                # 检查是否匹配删除规则
                matched_rule = None
                for rule in self.rules:
                    if rule.is_section_header and re.search(rule.pattern, line, re.IGNORECASE | re.MULTILINE):
                        matched_rule = rule
                        break

                if matched_rule:
                    # 匹配到需要删除的章节
                    skip_until_next_section = True
                    current_section_name = matched_rule.name
                    stats["removed_sections"].append({
                        "name": matched_rule.name,
                        "description": matched_rule.description,
                        "header": line.strip()
                    })
                    logger.debug(f"删除章节: {line.strip()}")
                    continue
                else:
                    # 新的有效章节，停止跳过
                    skip_until_next_section = False
                    current_section_name = None

            # 如果不在跳过状态，保留该行
            if not skip_until_next_section:
                result_lines.append(line)

        return '\n'.join(result_lines)

    def _remove_invalid_paragraphs(self, content: str, stats: Dict) -> str:
        """删除无效段落"""
        paragraphs = content.split('\n\n')
        cleaned_paragraphs = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 检查是否匹配段落级别的删除规则
            should_remove = False
            for rule in self.rules:
                if not rule.is_section_header and re.search(rule.pattern, para, re.IGNORECASE):
                    should_remove = True
                    stats["removed_paragraphs"] += 1
                    logger.debug(f"删除段落 ({rule.name}): {para[:100]}...")
                    break

            if not should_remove:
                cleaned_paragraphs.append(para)

        return '\n\n'.join(cleaned_paragraphs)

    def _merge_pages(self, pages: List[Tuple[int, str]]) -> str:
        """合并页面"""
        result_parts = []

        for page_num, content in pages:
            if page_num == 0:
                result_parts.append(content)
            else:
                result_parts.append(f"\n\n--- Page {page_num} ---\n\n{content}")

        return ''.join(result_parts)

    def get_valid_content_indicators(self) -> List[str]:
        """获取有效内容的指示词（用于判断是否保留）"""
        return [
            "Executive Summary",
            "Investment Thesis",
            "Key Takeaways",
            "Analysis",
            "Outlook",
            "Valuation",
            "Financial",
            "Results",
            "Performance",
            "Guidance",
            "Recommendation",
            "Conclusion",
            "Summary",
            "Overview",
            "Highlights",
            "Key Points",
            "Main Content",
            "Discussion",
            "Commentary"
        ]


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_md = """
# Company Analysis Report

## Executive Summary

This is the main analysis content that should be kept.

## Analyst Certification

I, John Doe, certify that the views expressed in this research report...

## Important Disclosures

Barclays Bank PLC has received compensation for investment banking services...

## United Kingdom

This document is being distributed in the United Kingdom...

## Americas

This document is being distributed in the Americas...

Bloomberg® is a trademark and service mark of Bloomberg Finance L.P.

© Copyright 2025 Barclays Bank PLC. All rights reserved.

--- Page 1 ---

## Financial Analysis

Revenue increased by 15% YoY...

## Disclaimer

This material is not intended for distribution to...
"""

    cleaner = MarkdownCleaner()
    cleaned, stats = cleaner.clean(test_md)

    print("=" * 60)
    print("清洗后的内容:")
    print("=" * 60)
    print(cleaned)
    print("\n" + "=" * 60)
    print("统计信息:")
    print("=" * 60)
    import json
    print(json.dumps(stats, indent=2, ensure_ascii=False))
