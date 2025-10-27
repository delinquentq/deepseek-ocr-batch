#!/usr/bin/env python3
"""
测试并行处理流程 - 验证阶段A和阶段B是否正确并行运行
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from batch_pdf_processor import BatchPDFProcessor

async def test_parallel_processing():
    """测试并行处理流程"""
    print("=" * 80)
    print("测试并行处理流程")
    print("=" * 80)

    # 选择3个测试PDF
    test_pdfs = [
        "input_pdfs/25.9月 普通个人版/9.26 普通个人版/Citi/Hans Laser Technology 002008.SZ Whats New at Citi Industrial SMID Transport Conf 2025 - Bullish PCB and IT Biz Whats New_2025-09-24.pdf",
        "input_pdfs/25.9月 普通个人版/9.26 普通个人版/Citi/Kingfisher KGF.L Model Update Model UpdateKingfisher KGF.L Model Update Model Update_2025-09-24.pdf",
        "input_pdfs/25.9月 普通个人版/9.26 普通个人版/Citi/Catalyst Watch Tracker September 17 23 Catalyst Watch Tracker September 17 23_2025-09-26.pdf",
    ]

    # 验证文件存在
    existing_pdfs = []
    for pdf in test_pdfs:
        if Path(pdf).exists():
            existing_pdfs.append(pdf)
            print(f"✓ 找到测试文件: {Path(pdf).name}")
        else:
            print(f"✗ 文件不存在: {pdf}")

    if not existing_pdfs:
        print("\n❌ 没有找到测试文件！")
        return

    print(f"\n将测试 {len(existing_pdfs)} 个PDF文件的并行处理")
    print("-" * 80)

    # 创建处理器
    processor = BatchPDFProcessor()

    # 运行批量处理
    print("\n开始批量处理...")
    results = await processor.process_batch(existing_pdfs)

    print("\n" + "=" * 80)
    print(f"测试完成！成功处理 {len(results)} 个文件")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_parallel_processing())
