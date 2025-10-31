#!/usr/bin/env python3
"""
修复错误日期的输出文件
从PDF元数据或文件修改时间中提取正确的日期
"""

import os
import re
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

BASE_DIR = Path(__file__).parent
OUTPUT_REPORT = BASE_DIR / "output_report"
OUTPUT_RESULTS = BASE_DIR / "output_results"
INPUT_PDFS = BASE_DIR / "input_pdfs"

# 错误的日期列表
WRONG_DATES = ["2025-12-31", "2025-11-30", "2025-11-01"]  # 未来日期，肯定是错的

def find_pdf_file(json_file: Path) -> Optional[Path]:
    """根据JSON文件名在input_pdfs中找到对应的PDF"""
    # 从JSON文件名中提取PDF名称（去掉.json和_final后缀）
    json_name = json_file.stem
    if json_name.endswith("_final"):
        json_name = json_name[:-6]

    # 尝试多种模式匹配
    patterns = [
        f"{json_name}*.pdf",
        f"{json_name.rsplit('_', 1)[0]}*.pdf",  # 去掉最后的部分
    ]

    for pattern in patterns:
        matches = list(INPUT_PDFS.rglob(pattern))
        if matches:
            return matches[0]

    return None

def extract_date_from_pdf(pdf_path: Path) -> Optional[str]:
    """从PDF元数据中提取日期"""
    if not HAS_PYMUPDF or not pdf_path.exists():
        return None

    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        doc.close()

        # 尝试从多个字段提取日期
        date_fields = ['creationDate', 'modDate', 'created', 'modified']
        for field in date_fields:
            if field in metadata and metadata[field]:
                date_str = metadata[field]
                # PDF日期格式: D:20250830...
                match = re.search(r'(\d{4})(\d{2})(\d{2})', date_str)
                if match:
                    year, month, day = match.groups()
                    try:
                        # 验证日期有效性
                        dt = datetime(int(year), int(month), int(day))
                        # 只接受2025年1月到10月的日期
                        if dt.year == 2025 and 1 <= dt.month <= 10:
                            return f"{year}-{month}-{day}"
                    except ValueError:
                        continue
    except Exception as e:
        print(f"  ❌ 读取PDF元数据失败: {e}")

    return None

def extract_date_from_filename(pdf_path: Path) -> Optional[str]:
    """从PDF文件名中提取日期（倒数第二个部分）"""
    # 文件名格式: xxx_2025-12-31.pdf
    # 但日期可能是错的，所以这个方法不可靠
    match = re.search(r'_(\d{4}-\d{2}-\d{2})\.pdf$', pdf_path.name)
    if match:
        date_str = match.group(1)
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            # 只接受合理的日期
            if dt.year == 2025 and 1 <= dt.month <= 10:
                return date_str
        except ValueError:
            pass
    return None

def get_file_modification_date(file_path: Path) -> str:
    """获取文件修改日期作为fallback"""
    mtime = os.path.getmtime(file_path)
    dt = datetime.fromtimestamp(mtime)
    return dt.strftime("%Y-%m-%d")

def fix_wrong_date_files(wrong_date: str, dry_run: bool = True):
    """修复指定错误日期下的所有文件"""
    wrong_date_dir = OUTPUT_REPORT / wrong_date

    if not wrong_date_dir.exists():
        print(f"⏭️  跳过不存在的目录: {wrong_date}")
        return

    json_files = list(wrong_date_dir.glob("*.json"))
    print(f"\n📁 处理目录: {wrong_date} ({len(json_files)} 个JSON文件)")

    fixed_count = 0
    failed_count = 0

    for json_file in json_files:
        print(f"\n  📄 处理: {json_file.name}")

        # 1. 找到对应的PDF文件
        pdf_file = find_pdf_file(json_file)
        if not pdf_file:
            print(f"    ❌ 未找到对应的PDF文件")
            failed_count += 1
            continue

        print(f"    ✅ 找到PDF: {pdf_file.name}")

        # 2. 提取正确的日期
        correct_date = None

        # 方法1: 从PDF元数据提取
        if HAS_PYMUPDF:
            correct_date = extract_date_from_pdf(pdf_file)
            if correct_date:
                print(f"    📅 从PDF元数据提取日期: {correct_date}")

        # 方法2: 从文件修改时间
        if not correct_date:
            correct_date = get_file_modification_date(pdf_file)
            print(f"    📅 使用文件修改日期: {correct_date}")

        # 3. 移动文件到正确的日期目录
        if correct_date and correct_date != wrong_date:
            correct_date_dir = OUTPUT_REPORT / correct_date

            if dry_run:
                print(f"    🔄 [DRY RUN] 将移动到: {correct_date_dir}/")
            else:
                # 创建目标目录
                correct_date_dir.mkdir(parents=True, exist_ok=True)

                # 移动文件
                target_path = correct_date_dir / json_file.name
                if target_path.exists():
                    print(f"    ⚠️  目标文件已存在，跳过: {target_path.name}")
                else:
                    shutil.move(str(json_file), str(target_path))
                    print(f"    ✅ 已移动到: {correct_date}/")
                    fixed_count += 1
        else:
            print(f"    ⏭️  日期未改变或无法确定正确日期")

    print(f"\n✅ 完成: 修复 {fixed_count} 个文件，失败 {failed_count} 个")

    # 如果不是dry run，且目录已空，删除错误日期目录
    if not dry_run and wrong_date_dir.exists():
        remaining = list(wrong_date_dir.glob("*.json"))
        if not remaining:
            wrong_date_dir.rmdir()
            print(f"🗑️  已删除空目录: {wrong_date}")

def main():
    import sys

    print("=" * 80)
    print("修复错误日期的输出文件")
    print("=" * 80)

    if not HAS_PYMUPDF:
        print("⚠️  警告: PyMuPDF未安装，无法从PDF元数据提取日期")
        print("    将使用文件修改时间作为替代方案")

    # 检查是否有 --yes 参数
    auto_confirm = "--yes" in sys.argv or "-y" in sys.argv

    # 首先dry run查看影响
    print("\n🔍 第一步：检查需要修复的文件（DRY RUN）")
    for wrong_date in WRONG_DATES:
        fix_wrong_date_files(wrong_date, dry_run=True)

    # 询问用户确认
    print("\n" + "=" * 80)
    if auto_confirm:
        response = "yes"
        print("自动确认执行修复（--yes参数）")
    else:
        try:
            response = input("是否执行修复？(yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "no"
            print("\n❌ 用户取消")

    if response in ['yes', 'y']:
        print("\n🔧 第二步：执行修复...")
        for wrong_date in WRONG_DATES:
            fix_wrong_date_files(wrong_date, dry_run=False)
        print("\n✅ 所有修复完成！")
    else:
        print("\n❌ 取消操作")

if __name__ == "__main__":
    main()
