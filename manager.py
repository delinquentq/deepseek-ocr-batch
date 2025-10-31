#!/usr/bin/env python3
"""
批量处理管理脚本 - 提供简单的命令行界面管理后台任务
"""

import subprocess
import sys
import time
import os
from pathlib import Path
import argparse

# 加载环境变量
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    print("警告: 未安装python-dotenv")
except Exception as e:
    print(f"加载.env文件失败: {e}")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
BATCH_SCRIPT = PROJECT_ROOT / "run_batch_auto.py"

def show_status():
    """显示当前运行状态"""
    cmd = [sys.executable, str(BATCH_SCRIPT), "--status"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("错误:", result.stderr)
    return result.returncode == 0

def start_daemon(base_dir="/home/qxx/report", filter_name=None, dry_run=False):
    """启动后台任务"""
    if dry_run:
        # 预览模式，不启动后台任务，直接运行预览
        cmd = [sys.executable, str(BATCH_SCRIPT), "--dry-run", "--base-dir", base_dir]
        if filter_name:
            cmd.extend(["--filter", filter_name])
    else:
        # 正常启动后台任务
        cmd = [sys.executable, str(BATCH_SCRIPT), "--daemon", "--base-dir", base_dir]
        if filter_name:
            cmd.extend(["--filter", filter_name])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("错误:", result.stderr)
    return result.returncode == 0

def stop_daemon():
    """停止后台任务"""
    cmd = [sys.executable, str(BATCH_SCRIPT), "--stop"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("错误:", result.stderr)
    return result.returncode == 0

def watch_logs(follow=True):
    """监控日志"""
    log_dir = PROJECT_ROOT / "logs"
    
    # 找最新的主日志文件
    log_files = list(log_dir.glob("auto_batch_*.log"))
    if not log_files:
        print("未找到日志文件")
        return
    
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    print(f"监控日志文件: {latest_log}")
    
    if follow:
        # 使用tail -f监控日志
        try:
            subprocess.run(["tail", "-f", str(latest_log)])
        except KeyboardInterrupt:
            print("\n停止监控")
    else:
        # 显示最后50行
        subprocess.run(["tail", "-50", str(latest_log)])

def show_recent_logs(lines=50):
    """显示最近的日志"""
    log_dir = PROJECT_ROOT / "logs"
    
    log_files = list(log_dir.glob("auto_batch_*.log"))
    if not log_files:
        print("未找到日志文件")
        return
    
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    subprocess.run(["tail", f"-{lines}", str(latest_log)])

def main():
    parser = argparse.ArgumentParser(
        description="批量处理管理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 启动后台处理
  python manager.py start

  # 启动后台处理（指定目录）
  python manager.py start --base-dir "/home/qxx/report"

  # 启动后台处理（过滤特定日期）
  python manager.py start --filter "10.1"

  # 查看状态
  python manager.py status

  # 停止处理
  python manager.py stop

  # 监控日志
  python manager.py logs

  # 查看最近日志
  python manager.py recent

  # 启动并监控
  python manager.py start --watch
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # start 命令
    start_parser = subparsers.add_parser("start", help="启动后台处理")
    start_parser.add_argument("--base-dir", default="/home/qxx/report", help="基础目录")
    start_parser.add_argument("--filter", help="过滤条件")
    start_parser.add_argument("--watch", action="store_true", help="启动后开始监控日志")
    start_parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际启动后台任务")

    # status 命令
    subparsers.add_parser("status", help="查看运行状态")

    # stop 命令
    subparsers.add_parser("stop", help="停止后台处理")

    # logs 命令
    logs_parser = subparsers.add_parser("logs", help="监控日志")
    logs_parser.add_argument("--no-follow", action="store_true", help="不跟踪实时日志")

    # recent 命令
    recent_parser = subparsers.add_parser("recent", help="查看最近日志")
    recent_parser.add_argument("--lines", type=int, default=50, help="显示行数")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "start":
        if start_daemon(args.base_dir, args.filter, args.dry_run):
            if args.watch and not args.dry_run:  # 预览模式不需要监控
                print("\n等待3秒后开始监控日志...")
                time.sleep(3)
                watch_logs(follow=True)
            return 0
        return 1

    elif args.command == "status":
        if show_status():
            return 0
        return 1

    elif args.command == "stop":
        if stop_daemon():
            return 0
        return 1

    elif args.command == "logs":
        watch_logs(follow=not args.no_follow)
        return 0

    elif args.command == "recent":
        show_recent_logs(args.lines)
        return 0

    return 0

if __name__ == "__main__":
    sys.exit(main())