#!/usr/bin/env python3
"""简单测试 - 只测试已有MD文件的阶段B处理"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from batch_pdf_processor import BatchPDFProcessor

async def test_simple():
    """测试已有MD文件的处理"""
    print("=" * 80)
    print("简单测试：处理已有MD文件")
    print("=" * 80)

    # 使用第一个测试中已经生成MD的文件
    test_pdf = "input_pdfs/25.9月 普通个人版/9.26 普通个人版/Citi/Hans Laser Technology 002008.SZ Whats New at Citi Industrial SMID Transport Conf 2025 - Bullish PCB and IT Biz Whats New_2025-09-24.pdf"

    if not Path(test_pdf).exists():
        print(f"❌ 文件不存在: {test_pdf}")
        return

    print(f"✓ 测试文件: {Path(test_pdf).name}")
    print("-" * 80)

    processor = BatchPDFProcessor()

    print("\n开始处理...")
    results = await processor.process_batch([test_pdf])

    print("\n" + "=" * 80)
    print(f"测试完成！结果: {len(results)} 个")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_simple())
