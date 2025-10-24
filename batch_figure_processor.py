#!/usr/bin/env python3
"""
批量图表处理器 - 适配新JSON Schema
一次性处理10-20张图表，大幅提升速度和降低API调用成本

输出格式：
{
  "id": "chart_0_0",
  "type": "bar|line|pie|table|area|scatter|waterfall|composed|gauge",
  "title": "图表标题",
  "confidence": 0.95,
  "data": {
    "labels": [...],  // 或 "columns" (表格)
    "datasets": [...]  // 或 "rows" (表格)
  },
  "options": {
    "source": "...",
    "height_estimate": 250,
    "page_break": "avoid"
  }
}
"""

import base64
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import re

logger = logging.getLogger(__name__)


class BatchFigureProcessor:
    """批量图表处理器 - 输出新Schema格式"""

    def __init__(self, batch_size: int = 15):
        """
        Args:
            batch_size: 每批处理的图片数量（建议10-20）
        """
        self.batch_size = batch_size

    def encode_images_batch(self, image_paths: List[str]) -> List[Dict]:
        """批量编码图片为base64"""
        encoded_images = []

        for idx, img_path in enumerate(image_paths):
            try:
                with open(img_path, 'rb') as f:
                    b64_data = base64.b64encode(f.read()).decode('utf-8')

                page_idx = self._infer_page_from_path(img_path)
                fig_idx = self._infer_figure_index_from_path(img_path)

                encoded_images.append({
                    "index": idx,
                    "path": img_path,
                    "page": page_idx,
                    "figure_index": fig_idx,
                    "base64": b64_data
                })
            except Exception as e:
                logger.error(f"编码图片失败 {img_path}: {e}")
                continue

        return encoded_images

    def _infer_page_from_path(self, image_path: str) -> int:
        """从路径推断页码"""
        try:
            name = Path(image_path).name
            page_str = name.split('_')[0]
            return int(page_str)  # 保持0-based（与chart_id一致）
        except Exception:
            return 0

    def _infer_figure_index_from_path(self, image_path: str) -> int:
        """从路径推断图表索引"""
        try:
            name = Path(image_path).stem
            fig_str = name.split('_')[1]
            return int(fig_str)
        except Exception:
            return 0

    async def process_figures_batch(self, processor, image_paths: List[str],
                                   semaphore: asyncio.Semaphore,
                                   markdown_content: str = None) -> List[Dict]:
        """
        批量处理图表

        Args:
            processor: OpenRouterProcessor实例
            image_paths: 图片路径列表
            semaphore: 并发控制信号量
            markdown_content: Markdown内容（用于提取图表标题和来源）

        Returns:
            图表数据列表（新Schema格式）
        """
        if not image_paths:
            return []

        logger.info(f"开始批量处理 {len(image_paths)} 张图表，批次大小: {self.batch_size}")

        # 提取图表上下文信息
        figure_contexts = self._extract_figure_contexts(markdown_content, image_paths) if markdown_content else {}

        all_results = []

        # 分批处理
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            logger.info(f"处理批次 {i // self.batch_size + 1}: {len(batch_paths)} 张图片")

            try:
                batch_results = await self._process_single_batch(
                    processor, batch_paths, semaphore, figure_contexts
                )
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"批次处理失败: {e}")
                # 降级：逐个处理失败的批次
                logger.info("降级为单张处理...")
                for path in batch_paths:
                    try:
                        single_result = await self._process_single_figure(
                            processor, path, semaphore, figure_contexts.get(path, {})
                        )
                        if single_result:
                            all_results.append(single_result)
                    except Exception as e2:
                        logger.error(f"单张处理也失败 {path}: {e2}")

        logger.info(f"✓ 批量处理完成，成功: {len(all_results)}/{len(image_paths)}")
        return all_results

    async def _process_single_batch(self, processor, batch_paths: List[str],
                                    semaphore: asyncio.Semaphore,
                                    figure_contexts: Dict[str, Dict] = None) -> List[Dict]:
        """处理单个批次"""
        async with semaphore:
            # 1. 编码所有图片
            encoded_images = self.encode_images_batch(batch_paths)
            if not encoded_images:
                return []

            # 2. 构建批量处理的prompt（新Schema格式）
            messages = self._build_batch_messages(encoded_images, figure_contexts or {})

            # 3. 调用API
            try:
                resp = await processor.call_model(
                    "gemini",
                    messages,
                    max_tokens=8192  # 批量处理需要更多tokens
                )
                content = resp['choices'][0]['message']['content']

                # 4. 解析批量返回的JSON（新Schema格式）
                charts_data = self._parse_batch_response(content, encoded_images)
                return charts_data

            except Exception as e:
                logger.error(f"批量API调用失败: {e}")
                raise

    def _build_batch_messages(self, encoded_images: List[Dict], figure_contexts: Dict[str, Dict]) -> List[Dict]:
        """构建批量处理的消息（新Schema格式）"""
        # 构建图片列表描述
        image_list_parts = []
        for img in encoded_images:
            img_path = img['path']
            context = figure_contexts.get(img_path, {})
            title = context.get('title', '')
            source = context.get('source', '')

            desc = f"- 图片{img['index'] + 1}: 第{img['page']}页，图表{img['figure_index']}"
            if title:
                desc += f"\n  标题: {title}"
            if source:
                desc += f"\n  来源: {source}"
            image_list_parts.append(desc)

        image_list = "\n".join(image_list_parts)

        # 构建content数组
        content_parts = [
            {
                "type": "text",
                "text": f"""请批量分析以下 {len(encoded_images)} 张图表，为每张图表提取数据并输出JSON数组。

图片列表：
{image_list}

对每张图表，提取以下信息并按照指定格式输出：

**输出格式（JSON数组）：**
[
  {{
    "image_index": 0,
    "type": "bar",  // 类型: bar/line/pie/table/area/scatter/waterfall/composed/gauge/stock_price
    "title": "Revenue Growth",
    "confidence": 0.95,  // 识别置信度 (0-1)
    "data": {{
      "labels": ["Q1", "Q2", "Q3", "Q4"],  // X轴标签（表格用"columns"）
      "datasets": [  // 数据集（表格用"rows"）
        {{
          "label": "Revenue",
          "values": [20, 30, 45, 60],
          "color": "#007bff"  // 可选
        }}
      ]
    }},
    "options": {{
      "source": "Company Reports",  // 数据来源
      "height_estimate": 250,  // 预估高度（像素）
      "page_break": "avoid"  // 分页控制
    }}
  }},
  {{
    "image_index": 1,
    ...
  }}
]

**重要说明：**
1. **type字段**：必须是以下之一：bar, line, pie, table, area, scatter, waterfall, composed, gauge, stock_price
2. **股价图识别**：如果图表是股价走势图（K线图、蜡烛图、股票价格曲线等），请设置type为"stock_price"，并将data设为空对象{{}}
3. **表格类型**：如果是table，data结构为：
   {{
     "columns": ["列1", "列2", "列3"],
     "rows": [
       {{"列1": "值1", "列2": "值2", "列3": "值3"}},
       ...
     ]
   }}
4. **confidence**：根据图表清晰度和识别准确度评估（0.0-1.0）
5. **labels**：X轴标签或类别名称
6. **datasets**：每个数据系列包含label、values、可选的color
7. **options.source**：如果已知来源，填入；否则留空字符串
8. **image_index**：对应图片顺序（从0开始）
9. 如果某张图无法识别，设置type为"gauge"，confidence为0.5，data为空结构

**股价图特征（请识别并跳过）：**
- K线图/蜡烛图（红绿柱状图）
- 股票价格走势曲线
- 带有成交量柱状图的价格图
- X轴为日期，Y轴为价格的时间序列图
- 包含"股价"、"Price"、"Stock"等关键词的图表

**仅输出JSON数组，不要其他文字。**

现在开始分析："""
            }
        ]

        # 添加所有图片
        for img in encoded_images:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img['base64']}"}
            })

        return [
            {
                "role": "system",
                "content": "你是专业的图表数据批量提取专家。请严格按照新的JSON Schema格式输出所有图表的数据。"
            },
            {
                "role": "user",
                "content": content_parts
            }
        ]

    def _parse_batch_response(self, response_text: str,
                              encoded_images: List[Dict]) -> List[Dict]:
        """解析批量返回的JSON（新Schema格式）"""
        try:
            # 尝试提取JSON数组
            charts_array = self._extract_json_array(response_text)

            if not charts_array:
                logger.warning("无法解析批量响应，返回空列表")
                return []

            # 转换为新Schema格式
            results = []
            for chart_data in charts_array:
                img_idx = chart_data.get("image_index", 0)

                # 查找对应的图片信息
                img_info = None
                for img in encoded_images:
                    if img["index"] == img_idx:
                        img_info = img
                        break

                if not img_info:
                    logger.warning(f"找不到image_index={img_idx}的图片信息")
                    continue

                # 构建新Schema格式的chart对象
                chart_id = f"chart_{img_info['page']}_{img_info['figure_index']}"

                # 确保type字段有效
                chart_type = chart_data.get("type", "bar")
                valid_types = ["bar", "line", "pie", "table", "area", "scatter", "waterfall", "composed", "gauge"]
                if chart_type not in valid_types:
                    chart_type = "bar"  # 默认为bar

                # 跳过股价图
                if chart_type == "stock_price":
                    logger.info(f"跳过股价图: {chart_id}")
                    continue

                # 生成正确的figure_id (hash格式)
                figure_id = hashlib.md5(f"{img_info['path']}_{img_info['page']}".encode()).hexdigest()[:16]

                # 构建v1.3.1格式的figure对象
                new_chart = {
                    "figure_id": figure_id,
                    "type": chart_type,
                    "title": chart_data.get("title", ""),
                    "page": img_info["page"] + 1,  # 转为1-based
                    "series": chart_data.get("data", {}).get("datasets", []),
                    "axes": chart_data.get("data", {}).get("axes", {}),
                    "provenance": {"page": img_info["page"] + 1}
                }

                # 添加source字段(从options或context中获取)
                source = chart_data.get("options", {}).get("source", "")
                if not source:
                    # 从context中获取
                    img_path = img_info['path']
                    context = figure_contexts.get(img_path, {})
                    source = context.get('source', '')
                if source:
                    new_chart["source"] = source

                # 如果没有axes,从data中提取
                if not new_chart["axes"] and "data" in chart_data:
                    data = chart_data["data"]
                    if "labels" in data:
                        new_chart["axes"] = {
                            "x": {"labels": data["labels"]},
                            "y": {}
                        }

                results.append(new_chart)

            return results

        except Exception as e:
            logger.error(f"解析批量响应失败: {e}")
            return []

    def _extract_json_array(self, response_text: str) -> Optional[List[Dict]]:
        """从响应中提取JSON数组"""
        # 策略1: 直接解析
        try:
            data = json.loads(response_text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 策略2: 提取markdown代码块中的JSON数组
        code_block_patterns = [
            r'```json\s*\n(\[[\s\S]*?\])\s*\n```',
            r'```\s*\n(\[[\s\S]*?\])\s*\n```',
        ]

        for pattern in code_block_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1).strip()
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析失败: {e}")
                    continue

        # 策略3: 查找第一个完整的JSON数组（括号匹配）
        bracket_count = 0
        start_idx = -1
        for i, char in enumerate(response_text):
            if char == '[':
                if bracket_count == 0:
                    start_idx = i
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0 and start_idx != -1:
                    try:
                        json_str = response_text[start_idx:i+1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        start_idx = -1
                        bracket_count = 0

        logger.error(f"无法提取JSON数组，响应前500字符: {response_text[:500]}")
        return None

    async def _process_single_figure(self, processor, image_path: str,
                                     semaphore: asyncio.Semaphore,
                                     context: Dict = None) -> Optional[Dict]:
        """处理单张图片（降级方案，新Schema格式）"""
        async with semaphore:
            try:
                with open(image_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')

                page_idx = self._infer_page_from_path(image_path)
                fig_idx = self._infer_figure_index_from_path(image_path)
                context = context or {}
                title_hint = context.get('title', '')
                source_hint = context.get('source', '')

                # 构建提示文本
                context_text = f"第{page_idx}页"
                if title_hint:
                    context_text += f"\n已知标题: {title_hint}"
                if source_hint:
                    context_text += f"\n已知来源: {source_hint}"

                messages = [
                    {
                        "role": "system",
                        "content": "你是专业的图表数据提取专家。请按照新的JSON Schema格式输出。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""请分析这张图表（{context_text}），提取数据并输出JSON。

输出格式：
{{
  "type": "bar",  // bar/line/pie/table/area/scatter/waterfall/composed/gauge/stock_price
  "title": "图表标题",
  "confidence": 0.95,
  "data": {{
    "labels": ["Q1", "Q2", "Q3", "Q4"],  // 表格用"columns"
    "datasets": [  // 表格用"rows"
      {{"label": "Revenue", "values": [20, 30, 45, 60], "color": "#007bff"}}
    ]
  }},
  "options": {{
    "source": "数据来源",
    "height_estimate": 250,
    "page_break": "avoid"
  }}
}}

**重要：**
- 如果是表格，data结构为: {{"columns": [...], "rows": [...]}}
- **如果是股价图（K线图、蜡烛图、股票价格走势等），设置type为"stock_price"，data为空对象{{}}**
- confidence为识别置信度（0-1）
- 如果已知标题和来源，请在JSON中包含

**股价图特征：**
- K线图/蜡烛图（红绿柱状图）
- 股票价格走势曲线
- 带有成交量柱状图的价格图
- X轴为日期，Y轴为价格

仅输出JSON，不要其他文字。"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                            }
                        ]
                    }
                ]

                resp = await processor.call_model("gemini", messages, max_tokens=4096)
                content = resp['choices'][0]['message']['content']

                # 提取JSON
                chart_data = self._extract_json_from_response(content)
                if chart_data:
                    chart_type = chart_data.get("type", "bar")

                    # 跳过股价图
                    if chart_type == "stock_price":
                        logger.info(f"跳过股价图: {image_path}")
                        return None

                    valid_types = ["bar", "line", "pie", "table", "area", "scatter", "waterfall", "composed", "gauge"]
                    if chart_type not in valid_types:
                        chart_type = "bar"

                    # 生成正确的figure_id (hash格式)
                    figure_id = hashlib.md5(f"{image_path}_{page_idx}".encode()).hexdigest()[:16]

                    new_chart = {
                        "figure_id": figure_id,
                        "type": chart_type,
                        "title": chart_data.get("title", title_hint or ""),
                        "page": page_idx + 1,  # 转为1-based
                        "series": chart_data.get("data", {}).get("datasets", []),
                        "axes": chart_data.get("data", {}).get("axes", {}),
                        "provenance": {"page": page_idx + 1}
                    }

                    # 添加source字段
                    source = chart_data.get("options", {}).get("source", "") or source_hint
                    if source:
                        new_chart["source"] = source

                    # 如果没有axes,从data中提取
                    if not new_chart["axes"] and "data" in chart_data:
                        data = chart_data["data"]
                        if "labels" in data:
                            new_chart["axes"] = {
                                "x": {"labels": data["labels"]},
                                "y": {}
                            }

                    return new_chart

            except Exception as e:
                logger.error(f"单张图片处理失败 {image_path}: {e}")
                return None

    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """从响应中提取JSON对象"""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # 尝试提取代码块
        match = re.search(r'```json\s*\n(\{.*?\})\s*\n```', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 查找第一个完整的JSON对象
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(response_text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        return json.loads(response_text[start_idx:i+1])
                    except json.JSONDecodeError:
                        start_idx = -1
                        brace_count = 0

        return None

    def _extract_figure_contexts(self, markdown_content: str, image_paths: List[str]) -> Dict[str, Dict]:
        """
        从Markdown中提取图表的标题和来源信息

        Args:
            markdown_content: Markdown文本内容
            image_paths: 图片路径列表

        Returns:
            {image_path: {"title": "...", "source": "..."}}
        """
        contexts = {}

        if not markdown_content:
            return contexts

        # 为每个图片路径提取上下文
        for img_path in image_paths:
            # 从路径中提取图片文件名，例如 "0_0.jpg"
            img_filename = Path(img_path).name

            # 在Markdown中查找图片引用
            # 格式: ![Figure X-Y](images/X_Y.jpg)
            pattern = rf'!\[Figure[^\]]*\]\(images/{re.escape(img_filename)}\)'
            match = re.search(pattern, markdown_content)

            if match:
                # 提取图片引用后的标题和来源
                start_pos = match.end()
                # 查找接下来的500个字符
                context_text = markdown_content[start_pos:start_pos + 500]

                # 提取标题 (FIGURE X. ...)
                title_match = re.search(r'<center>\s*FIGURE\s+\d+\.\s*([^<]+?)\s*</center>', context_text, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else None

                # 提取来源 (Source: ...)
                source_match = re.search(r'<center>\s*([^<]*?Source:\s*[^<]+?)\s*</center>', context_text, re.IGNORECASE)
                source = source_match.group(1).strip() if source_match else None

                if title or source:
                    contexts[img_path] = {
                        "title": title,
                        "source": source
                    }
                    logger.info(f"提取图表上下文: {img_filename} - 标题: {title}, 来源: {source}")

        return contexts


# 测试代码
if __name__ == "__main__":
    import asyncio

    async def test():
        processor = BatchFigureProcessor(batch_size=10)
        print("BatchFigureProcessor 已创建（新Schema格式），批次大小: 10")

    asyncio.run(test())
