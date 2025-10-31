#!/usr/bin/env python3
"""
图像过滤器
负责识别并移除无价值的图像（例如股价走势图、披露页图像），避免浪费大模型配额
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

@dataclass(frozen=True)
class FigureCandidate:
    """Markdown中出现的图像引用"""

    page: int
    label: str
    relative_path: str
    context: str

    @property
    def filename(self) -> str:
        return Path(self.relative_path).name


class FigureFilter:
    """根据上下文过滤股价图、披露页图像等无效图片"""

    _FIGURE_PATTERN = re.compile(r"!\[(?P<label>[^\]]*)\]\((?P<path>[^)]+)\)", re.IGNORECASE)
    _PAGE_SPLIT_PATTERN = re.compile(r"\n\n--- Page (\d+) ---\n\n")

    _PRICE_KEYWORDS = re.compile(
        r"(rating|price|target|history|performance|relative|total return|12[- ]?month)",
        re.IGNORECASE,
    )
    _INDEX_KEYWORDS = re.compile(
        r"(s&p|msci|sox|nasdaq|dow|ftse|hang\s*seng|hsi|spx|ndx|tsx)", re.IGNORECASE
    )
    _TIME_KEYWORDS = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|20\d{2})",
                                re.IGNORECASE)

    def __init__(self) -> None:
        pass

    def filter(
        self,
        raw_markdown: str,
        figure_paths: Sequence[str],
    ) -> Tuple[List[str], List[str]]:
        """
        根据Markdown上下文过滤图像

        Args:
            raw_markdown: OCR生成的原始Markdown
            figure_paths: 阶段A生成的图像绝对路径序列

        Returns:
            (保留的图像路径列表, 被移除的图像文件名列表)
        """
        if not figure_paths:
            return [], []

        candidates = self._collect_candidates(raw_markdown)
        if not candidates:
            return list(figure_paths), []

        drop_names = {c.filename for c in candidates if self._should_drop(c)}

        retained = [path for path in figure_paths if Path(path).name not in drop_names]
        dropped = sorted(drop_names)
        return retained, dropped

    def _collect_candidates(self, markdown: str) -> List[FigureCandidate]:
        """收集所有图像及其上下文"""
        candidates: List[FigureCandidate] = []
        parts = re.split(self._PAGE_SPLIT_PATTERN, markdown)

        page_contents: List[Tuple[int, str]] = []
        if parts and parts[0].strip():
            page_contents.append((0, parts[0]))
        for idx in range(1, len(parts), 2):
            if idx + 1 >= len(parts):
                break
            page_num = int(parts[idx])
            page_contents.append((page_num, parts[idx + 1]))

        for page_num, content in page_contents:
            for match in self._FIGURE_PATTERN.finditer(content):
                label = match.group("label") or ""
                rel_path = match.group("path")
                context = self._extract_context(content, match.start(), match.end())
                candidates.append(
                    FigureCandidate(
                        page=page_num,
                        label=label.strip(),
                        relative_path=rel_path.strip(),
                        context=context,
                    )
                )

        return candidates

    def _should_drop(self, candidate: FigureCandidate) -> bool:
        """判断图像是否应该被过滤"""
        text_blob = f"{candidate.label}\n{candidate.context}"

        if self._page_is_disclosure(text_blob):
            return True

        if self._looks_like_price_chart(text_blob):
            return True

        return False

    def _page_is_disclosure(self, text: str) -> bool:
        """检测文本是否属于披露/法律等模板内容"""
        lowered = text.lower()
        indicators = (
            "analyst(s) certification",
            "important disclosure",
            "availability of disclosure",
            "risk disclosure",
            "disclosure legend",
            "legal entities involved in producing",
            "united kingdom",
            "hong kong",
            "middle east",
            "australia",
            "singapore",
            "americas",
            "bloomberg® is a trademark",
            "barclays bank plc",
            "barclays capital inc",
        )
        return any(indicator.lower() in lowered for indicator in indicators)

    def _looks_like_price_chart(self, text: str) -> bool:
        """检测是否符合股价走势图的关键信息"""
        if not self._PRICE_KEYWORDS.search(text):
            return False
        if not self._INDEX_KEYWORDS.search(text):
            return False
        if not self._TIME_KEYWORDS.search(text):
            return False
        return True

    @staticmethod
    def _extract_context(text: str, start: int, end: int, window: int = 240) -> str:
        """截取图像引用附近文本，便于做规则判断"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end]
        context = re.sub(r"\s+", " ", context)
        return context.strip()


def filter_figures(markdown: str, figure_paths: Iterable[str]) -> Tuple[List[str], List[str]]:
    """
    便捷方法：过滤图像
    """
    return FigureFilter().filter(markdown, list(figure_paths))
