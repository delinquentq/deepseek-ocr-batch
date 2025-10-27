#!/usr/bin/env python3
"""
修复 output_results 中错误日期的文件夹
从对应的PDF文件中提取正确的日期
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
OUTPUT_RESULTS = BASE_DIR / "output_results"
INPUT_PDFS = BASE_DIR / "input_pdfs"

# 错误的日期列表
WRONG_DATES = ["2025-12-31", "2025-11-30", "2025-11-01"]

def find_pdf_for_folder(folder_name: str) -> Path:
    """根据文件夹名在input_pdfs中找到对应的PDF"""
    # 尝试直接匹配
    matches = list(INPUT_PDFS.rglob(f"{folder_name}*.pdf"))
    if matches:
        return matches[0]

    # 尝试部分匹配
    parts = folder_name.split('_')
    if len(parts) > 1:
        pattern = parts[0] + "*.pdf"
        matches = list(INPUT_PDFS.rglob(pattern))
        if matches:
            return matches[0]

    return None

def get_file_modification_date(file_path: Path) -> str:
    """获取文件修改日期"""
    mtime = os.path.getmtime(file_path)
    dt = datetime.fromtimestamp(mtime)
    return dt.strftime("%Y-%m-%d")

def fix_output_results_dates(wrong_date: str, dry_run: bool = True):
    """修复指定错误日期下的所有文件夹"""
    wrong_date_dir = OUTPUT_RESULTS / wrong_date

    if not wrong_date_dir.exists():
        print(f"⏭️  跳过不存在的目录: {wrong_date}")
        return

    # 获取所有子文件夹
    folders = [f for f in wrong_date_dir.iterdir() if f.is_dir()]
    print(f"\n📁 处理目录: {wrong_date} ({len(folders)} 个文件夹)")

    fixed_count = 0
    failed_count = 0

    for folder in folders:
        folder_name = folder.name
        print(f"\n  📂 处理: {folder_name}")

        # 查找对应的PDF
        pdf_file = find_pdf_for_folder(folder_name)

        if not pdf_file:
            print(f"    ❌ 未找到对应的PDF文件")
            failed_count += 1
            continue

        print(f"    ✅ 找到PDF: {pdf_file.name}")

        # 获取正确的日期
        correct_date = get_file_modification_date(pdf_file)
        print(f"    📅 正确日期: {correct_date}")

        if correct_date != wrong_date:
            correct_date_dir = OUTPUT_RESULTS / correct_date
            target_folder = correct_date_dir / folder_name

            if dry_run:
                print(f"    🔄 [DRY RUN] 将移动到: {correct_date_dir}/")
            else:
                # 创建目标日期目录
                correct_date_dir.mkdir(parents=True, exist_ok=True)

                # 移动整个文件夹
                if target_folder.exists():
                    print(f"    ⚠️  目标文件夹已存在，跳过")
                else:
                    shutil.move(str(folder), str(target_folder))
                    print(f"    ✅ 已移动到: {correct_date}/")
                    fixed_count += 1
        else:
            print(f"    ⏭️  日期未改变")

    print(f"\n✅ 完成: 修复 {fixed_count} 个文件夹，失败 {failed_count} 个")

    # 如果不是dry run，且目录已空，删除错误日期目录
    if not dry_run and wrong_date_dir.exists():
        remaining = list(wrong_date_dir.iterdir())
        if not remaining:
            wrong_date_dir.rmdir()
            print(f"🗑️  已删除空目录: {wrong_date}")

def main():
    import sys

    print("=" * 80)
    print("修复 output_results 中错误日期的文件夹")
    print("=" * 80)

    # 检查是否有 --yes 参数
    auto_confirm = "--yes" in sys.argv or "-y" in sys.argv

    # 首先dry run查看影响
    print("\n🔍 第一步：检查需要修复的文件夹（DRY RUN）")
    for wrong_date in WRONG_DATES:
        fix_output_results_dates(wrong_date, dry_run=True)

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
            fix_output_results_dates(wrong_date, dry_run=False)
        print("\n✅ 所有修复完成！")
    else:
        print("\n❌ 取消操作")

if __name__ == "__main__":
    main()
