#!/usr/bin/env python3
"""
JSON合并引擎
将规则引擎生成的基础JSON和视觉模型识别的图表数据智能合并

核心功能：
1. 合并figures数据到正确位置
2. 更新extraction_summary统计
3. 关联图表与文本段落
4. 验证合并后的数据完整性
"""

import logging
from typing import Dict, List
from copy import deepcopy

logger = logging.getLogger(__name__)


class JsonMerger:
    """JSON合并引擎"""

    def merge(self, base_json: Dict, figures_data: List[Dict]) -> Dict:
        """
        合并基础JSON和图表数据

        Args:
            base_json: 规则引擎生成的基础JSON
            figures_data: 视觉模型识别的图表数据列表

        Returns:
            合并后的完整JSON
        """
        logger.info(f"开始合并JSON: {len(figures_data)} 个图表")

        # 深拷贝避免修改原始数据
        result = deepcopy(base_json)

        # 1. 合并figures数据
        if "data" not in result:
            result["data"] = {}

        result["data"]["figures"] = figures_data

        # 2. 更新extraction_summary
        if "extraction_summary" not in result["data"]:
            result["data"]["extraction_summary"] = {}

        summary = result["data"]["extraction_summary"]
        summary["figures_count"] = len(figures_data)

        # 统计有完整数据的图表数量
        figures_with_data = sum(
            1 for fig in figures_data
            if fig.get("series") and len(fig.get("series", [])) > 0
        )
        summary["figures_with_data"] = figures_with_data

        # 3. 关联图表与段落（可选）
        self._link_figures_to_passages(result, figures_data)

        # 4. 更新模型信息
        if "doc" in result and "extraction_run" in result["doc"]:
            pipeline_steps = result["doc"]["extraction_run"].get("pipeline_steps", [])
            if "figure_vision" not in pipeline_steps:
                pipeline_steps.append("figure_vision")
            result["doc"]["extraction_run"]["pipeline_steps"] = pipeline_steps

            # 更新synthesis_model信息
            result["doc"]["extraction_run"]["synthesis_model"] = "hybrid:rule-engine+gemini-2.5-flash"

        logger.info(f"✓ JSON合并完成: {len(figures_data)} 个图表已整合")
        return result

    def _link_figures_to_passages(self, result: Dict, figures_data: List[Dict]):
        """关联图表与文本段落"""
        if "passages" not in result or not result["passages"]:
            return

        # 为每个图表查找同页的段落
        for figure in figures_data:
            figure_page = figure.get("page", 0)
            figure_title = figure.get("title") or ""
            if figure_title:
                figure_title = figure_title.lower()

            # 查找同页段落
            related_passages = []
            for passage in result["passages"]:
                if passage.get("page") == figure_page:
                    related_passages.append(passage["passage_id"])

            # 添加关联信息（可选字段）
            if related_passages:
                if "metadata" not in figure:
                    figure["metadata"] = {}
                figure["metadata"]["related_passages"] = related_passages[:3]  # 最多3个

    def validate_merged_json(self, merged_json: Dict) -> tuple[bool, List[str]]:
        """
        验证合并后的JSON

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 检查必需的顶层字段
        required_top_level = ["schema_version", "doc", "passages", "entities", "data"]
        for field in required_top_level:
            if field not in merged_json:
                errors.append(f"缺少顶层字段: {field}")

        # 检查data字段
        if "data" in merged_json:
            data = merged_json["data"]
            required_data_fields = ["figures", "tables", "numerical_data", "extraction_summary"]
            for field in required_data_fields:
                if field not in data:
                    errors.append(f"data缺少字段: {field}")

            # 检查figures
            if "figures" in data:
                for idx, figure in enumerate(data["figures"]):
                    if "figure_id" not in figure:
                        errors.append(f"figures[{idx}]缺少figure_id")
                    if "page" not in figure:
                        errors.append(f"figures[{idx}]缺少page")
                    if "type" not in figure:
                        errors.append(f"figures[{idx}]缺少type")

        # 检查doc字段
        if "doc" in merged_json:
            doc = merged_json["doc"]
            required_doc_fields = ["doc_id", "title", "timestamps", "extraction_run"]
            for field in required_doc_fields:
                if field not in doc:
                    errors.append(f"doc缺少字段: {field}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_merge_statistics(self, merged_json: Dict) -> Dict:
        """获取合并统计信息"""
        stats = {
            "figures_count": 0,
            "tables_count": 0,
            "numerical_data_count": 0,
            "passages_count": 0,
            "entities_count": 0,
            "pages_processed": 0
        }

        if "data" in merged_json:
            data = merged_json["data"]
            stats["figures_count"] = len(data.get("figures", []))
            stats["tables_count"] = len(data.get("tables", []))
            stats["numerical_data_count"] = len(data.get("numerical_data", []))

        stats["passages_count"] = len(merged_json.get("passages", []))
        stats["entities_count"] = len(merged_json.get("entities", []))

        if "doc" in merged_json and "extraction_run" in merged_json["doc"]:
            metadata = merged_json["doc"]["extraction_run"].get("processing_metadata", {})
            stats["pages_processed"] = metadata.get("pages_processed", 0)

        return stats


class FigurePositionMatcher:
    """图表位置匹配器 - 将图表插入到正确的文本位置"""

    def match_figures_to_markdown_positions(self, markdown_content: str,
                                           figures_data: List[Dict]) -> Dict[int, List[Dict]]:
        """
        匹配图表在Markdown中的位置

        Args:
            markdown_content: Markdown文本
            figures_data: 图表数据列表

        Returns:
            {page_num: [figures_on_this_page]}
        """
        import re

        # 按页分组图表
        figures_by_page = {}
        for figure in figures_data:
            page = figure.get("page", 0)
            if page not in figures_by_page:
                figures_by_page[page] = []
            figures_by_page[page].append(figure)

        # 查找Markdown中的图片引用位置
        # 格式: ![Figure X-Y](images/X_Y.jpg)
        figure_refs = re.finditer(r'!\[Figure (\d+)-(\d+)\]\(images/\d+_\d+\.jpg\)', markdown_content)

        ref_positions = {}
        for match in figure_refs:
            page_idx = int(match.group(1))
            fig_idx = int(match.group(2))
            position = match.start()

            if page_idx not in ref_positions:
                ref_positions[page_idx] = []
            ref_positions[page_idx].append({
                "fig_idx": fig_idx,
                "position": position
            })

        return figures_by_page, ref_positions


# 测试代码
if __name__ == "__main__":
    # 测试合并功能
    base_json = {
        "schema_version": "1.3.1",
        "doc": {
            "doc_id": "test123",
            "title": "Test Document",
            "timestamps": {},
            "extraction_run": {
                "pipeline_steps": ["ocr", "rule_extraction"]
            }
        },
        "passages": [
            {"passage_id": "p1", "text": "Test passage", "page": 1}
        ],
        "entities": [],
        "data": {
            "tables": [],
            "numerical_data": [],
            "extraction_summary": {
                "tables_count": 0,
                "numerical_data_count": 0
            }
        }
    }

    figures_data = [
        {
            "figure_id": "fig1",
            "type": "bar",
            "title": "Test Chart",
            "page": 1,
            "series": [{"name": "Series1", "values": [1, 2, 3]}],
            "provenance": {"page": 1}
        }
    ]

    merger = JsonMerger()
    result = merger.merge(base_json, figures_data)

    is_valid, errors = merger.validate_merged_json(result)
    print(f"验证结果: {'通过' if is_valid else '失败'}")
    if errors:
        print("错误:", errors)

    stats = merger.get_merge_statistics(result)
    print("统计:", stats)

    import json
    print("\n合并后的JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
