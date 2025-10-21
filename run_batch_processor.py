#!/usr/bin/env python3
"""
批量PDF处理系统启动脚本
简化版启动界面，方便用户使用
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import List

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from batch_pdf_processor import BatchPDFProcessor
    from config_batch import Config, setup_environment
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装所有依赖包: pip install -r requirements_batch.txt")
    sys.exit(1)

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    MAGENTA = '\033[35m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_banner():
    """打印启动横幅"""
    banner = f"""
{Colors.CYAN}{'='*80}
{Colors.BOLD}                    DeepSeek OCR 批量处理系统 v2.0
{Colors.CYAN}{'='*80}
{Colors.GREEN}🚀 基于 DeepSeek-OCR + OpenRouter 双模型的智能文档处理系统
{Colors.BLUE}🎯 针对 RTX 3090 24G 显存优化
{Colors.YELLOW}📊 支持批量PDF处理并输出结构化JSON数据
{Colors.CYAN}{'='*80}{Colors.RESET}
    """
    print(banner)

def print_system_info():
    """打印系统信息"""
    try:
        import torch
        print(f"{Colors.BLUE}🔧 系统信息:{Colors.RESET}")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA可用: {'✅ 是' if torch.cuda.is_available() else '❌ 否'}")

        if torch.cuda.is_available():
            print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print()
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️  系统信息获取失败: {e}{Colors.RESET}\n")

def check_environment():
    """检查环境配置"""
    print(f"{Colors.BLUE}🔍 环境检查:{Colors.RESET}")

    # 检查API密钥
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print(f"   OpenRouter API: ✅ 已配置 (...{api_key[-4:]})")
    else:
        print(f"   OpenRouter API: ❌ 未配置")
        print(f"   {Colors.YELLOW}请设置环境变量: export OPENROUTER_API_KEY=your_key{Colors.RESET}")
        return False

    # 检查Schema文件
    schema_path = Path("json schema.json")
    if schema_path.exists():
        print(f"   JSON Schema: ✅ 已找到")
    else:
        print(f"   JSON Schema: ❌ 未找到 ({schema_path})")
        return False

    # 检查目录
    config = Config()
    if config.paths.INPUT_DIR.exists():
        pdf_count = len(list(config.paths.INPUT_DIR.glob("*.pdf")))
        print(f"   输入目录: ✅ 找到 {pdf_count} 个PDF文件")
        if pdf_count == 0:
            print(f"   {Colors.YELLOW}请将PDF文件放入 {config.paths.INPUT_DIR} 目录{Colors.RESET}")
    else:
        print(f"   输入目录: ⚠️  将自动创建 ({config.paths.INPUT_DIR})")

    print()
    return True

def list_pdf_files(input_dir: Path) -> List[Path]:
    """列出PDF文件"""
    pdf_files = list(input_dir.glob("*.pdf"))
    return sorted(pdf_files)

def display_file_list(pdf_files: List[Path]):
    """显示文件列表"""
    print(f"{Colors.BLUE}📁 发现的PDF文件 ({len(pdf_files)} 个):{Colors.RESET}")

    if not pdf_files:
        print(f"   {Colors.YELLOW}暂无PDF文件{Colors.RESET}")
        return

    for i, pdf_file in enumerate(pdf_files, 1):
        size_mb = pdf_file.stat().st_size / 1024 / 1024
        print(f"   {i:2d}. {pdf_file.name} ({size_mb:.1f} MB)")
    print()

def get_user_confirmation(pdf_files: List[Path]) -> bool:
    """获取用户确认"""
    if not pdf_files:
        print(f"{Colors.YELLOW}请将PDF文件放入 input_pdfs 目录后重新运行{Colors.RESET}")
        return False

    print(f"{Colors.MAGENTA}⚡ 处理预估:{Colors.RESET}")
    print(f"   📊 预计处理时间: {len(pdf_files) * 2:.0f}-{len(pdf_files) * 5:.0f} 分钟")
    print(f"   💰 预计API费用: ${len(pdf_files) * 0.05:.2f}-${len(pdf_files) * 0.15:.2f}")
    print(f"   📈 显存使用: ~18-20 GB")
    print()

    try:
        response = input(f"{Colors.CYAN}是否开始批量处理? (y/N): {Colors.RESET}").strip().lower()
        return response in ['y', 'yes', '是']
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}用户取消操作{Colors.RESET}")
        return False

async def run_processing(args):
    """运行处理流程"""
    try:
        # 环境设置
        setup_environment()

        # 获取PDF文件列表
        config = Config()
        pdf_files = list_pdf_files(config.paths.INPUT_DIR)

        if args.file:
            # 处理指定文件
            specified_files = []
            for file_pattern in args.file:
                matching_files = [f for f in pdf_files if file_pattern in f.name]
                specified_files.extend(matching_files)
            pdf_files = specified_files

        if not pdf_files:
            print(f"{Colors.RED}❌ 没有找到要处理的PDF文件{Colors.RESET}")
            return False

        # 显示文件列表
        display_file_list(pdf_files)

        # 获取用户确认
        if not args.yes and not get_user_confirmation(pdf_files):
            return False

        # 开始处理
        print(f"{Colors.GREEN}🚀 开始批量处理...{Colors.RESET}\n")

        processor = BatchPDFProcessor()
        results = await processor.process_batch([str(f) for f in pdf_files])

        # 显示结果
        print(f"\n{Colors.GREEN}{'='*60}")
        print(f"✅ 批量处理完成!")
        print(f"{'='*60}{Colors.RESET}")
        print(f"📊 成功处理: {len(results)} 个文件")
        print(f"📁 输出目录: {config.paths.OUTPUT_DIR}")
        print(f"📝 日志文件: {config.paths.LOG_FILE}")

        return True

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  用户中断处理{Colors.RESET}")
        return False
    except Exception as e:
        print(f"\n{Colors.RED}❌ 处理失败: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DeepSeek OCR 批量处理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_batch_processor.py                    # 处理所有PDF文件
  python run_batch_processor.py -f report1.pdf    # 处理指定文件
  python run_batch_processor.py -y                # 跳过确认直接处理
  python run_batch_processor.py --setup           # 仅检查环境配置

注意事项:
  1. 请确保设置了 OPENROUTER_API_KEY 环境变量
  2. 将PDF文件放入 input_pdfs 目录
  3. 确保有足够的显存 (推荐RTX 3090 24G或更高)
        """
    )

    parser.add_argument('-f', '--file', action='append', metavar='FILENAME',
                       help='指定要处理的PDF文件名 (可多次使用)')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='跳过确认对话，直接开始处理')
    parser.add_argument('--setup', action='store_true',
                       help='仅检查环境配置，不进行处理')
    parser.add_argument('--version', action='version', version='%(prog)s 2.0')

    args = parser.parse_args()

    # 打印横幅
    print_banner()
    print_system_info()

    # 检查环境
    if not check_environment():
        print(f"{Colors.RED}❌ 环境检查失败，请修复上述问题后重试{Colors.RESET}")
        sys.exit(1)

    # 如果只是检查环境配置
    if args.setup:
        print(f"{Colors.GREEN}✅ 环境配置正常！{Colors.RESET}")
        return

    # 运行处理流程
    try:
        success = asyncio.run(run_processing(args))
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"{Colors.RED}❌ 启动失败: {e}{Colors.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()