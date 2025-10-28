#!/usr/bin/env python3
"""
自动批量处理脚本 - 按日期目录顺序处理PDF
支持自定义输入目录、并行处理、失败重试等功能
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import logging
import signal
import atexit
import json

# 加载环境变量
try:
    from dotenv import load_dotenv
    # 查找.env文件（当前目录和上级目录）
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"已加载环境变量: {env_file}")
    else:
        print("未找到.env文件")
except ImportError:
    print("警告: 未安装python-dotenv，请运行: pip install python-dotenv")
except Exception as e:
    print(f"加载.env文件失败: {e}")

# 配置日志
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# 全局变量用于状态管理
STATUS_FILE = LOG_DIR / "batch_status.json"
current_process = None
start_time = None

def save_status(status, message="", current_dir="", processed=0, total=0, failed=0):
    """保存处理状态到文件"""
    status_data = {
        "status": status,  # running, completed, stopped, error
        "message": message,
        "current_directory": current_dir,
        "processed": processed,
        "total": total,
        "failed": failed,
        "start_time": start_time.isoformat() if start_time else None,
        "last_update": datetime.now().isoformat()
    }
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"保存状态失败: {e}")

def signal_handler(signum, frame):
    """信号处理器，处理中断信号"""
    print(f"\n收到信号 {signum}，正在优雅停止...")
    if current_process:
        print(f"等待当前子进程结束...")
        current_process.terminate()
        try:
            current_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            current_process.kill()
    
    save_status("stopped", f"收到信号 {signum}，程序停止")
    print("状态已保存，程序退出")
    sys.exit(0)

def cleanup():
    """清理函数，在程序退出时调用"""
    if current_process and current_process.poll() is None:
        current_process.terminate()
        try:
            current_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            current_process.kill()

# 注册信号处理器和清理函数
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup)

def setup_logging(log_file):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def find_date_directories(base_dir):
    """查找所有日期目录"""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {base_dir}")

    # 获取所有子目录并排序
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    subdirs.sort()  # 按名称排序（如：9.1, 9.2, 9.3）

    return subdirs

def process_directory(input_dir, logger, dry_run=False):
    """处理单个目录"""
    global current_process
    
    dir_name = input_dir.name

    # 生成日志文件名
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in dir_name)
    log_file = LOG_DIR / f"run_{safe_name}_{datetime.now().strftime('%Y-%m-%d_%H%M')}.log"

    logger.info("=" * 60)
    logger.info(f"开始处理: {dir_name}")
    logger.info(f"输入路径: {input_dir}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN] 跳过实际处理")
        return True, "dry_run"

    # 构建命令
    cmd = [
        sys.executable,  # 使用当前Python解释器
        "run_batch_processor.py",
        "-y",
        "--input",
        str(input_dir)
    ]

    # 执行命令
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            # 使用Popen以便能够控制进程
            current_process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=Path(__file__).parent
            )
            
            # 等待进程完成
            exit_code = current_process.wait()
            current_process = None

        if exit_code == 0:
            logger.info(f"✓ 完成: {dir_name} (退出码: {exit_code})")
            return True, exit_code
        else:
            logger.error(f"✗ 失败: {dir_name} (退出码: {exit_code})")
            logger.error(f"  查看详细日志: {log_file}")
            return False, exit_code

    except Exception as e:
        logger.error(f"✗ 异常: {dir_name} - {str(e)}")
        logger.error(f"  查看详细日志: {log_file}")
        return False, str(e)
    finally:
        current_process = None
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    global start_time
    
    parser = argparse.ArgumentParser(
        description="自动批量处理PDF文档 - 按日期目录顺序处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认目录
  python run_batch_auto.py

  # 指定自定义基础目录
  python run_batch_auto.py --base-dir "/home/qxx/report"

  # 预览模式（不实际处理）
  python run_batch_auto.py --dry-run

  # 后台运行模式
  python run_batch_auto.py --daemon

  # 失败后继续处理其他目录
  python run_batch_auto.py --continue-on-error

  # 查看运行状态
  python run_batch_auto.py --status

  # 停止后台运行
  python run_batch_auto.py --stop
        """
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/qxx/report",
        help="基础输入目录（默认: /home/qxx/report）"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，仅列出要处理的目录，不实际执行"
    )

    parser.add_argument(
        "--daemon",
        action="store_true",
        help="后台运行模式（守护进程）"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="查看当前运行状态"
    )

    parser.add_argument(
        "--stop",
        action="store_true",
        help="停止后台运行的任务"
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="遇到失败时继续处理其他目录（默认启用）"
    )

    parser.add_argument(
        "--filter",
        help="只处理包含指定关键词的目录（例如：--filter '10.1'）"
    )

    args = parser.parse_args()

    # 处理状态查看和停止命令
    if args.status:
        try:
            if STATUS_FILE.exists():
                with open(STATUS_FILE, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
                
                print("=" * 60)
                print("批量处理状态:")
                print(f"状态: {status_data.get('status', 'unknown')}")
                print(f"消息: {status_data.get('message', '')}")
                print(f"当前目录: {status_data.get('current_directory', '')}")
                print(f"已处理: {status_data.get('processed', 0)}")
                print(f"总目录: {status_data.get('total', 0)}")
                print(f"失败: {status_data.get('failed', 0)}")
                print(f"开始时间: {status_data.get('start_time', '')}")
                print(f"最后更新: {status_data.get('last_update', '')}")
                print("=" * 60)
                return 0
            else:
                print("未找到状态文件，可能没有正在运行的任务")
                return 1
        except Exception as e:
            print(f"读取状态失败: {e}")
            return 1

    if args.stop:
        try:
            if STATUS_FILE.exists():
                with open(STATUS_FILE, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
                
                if status_data.get('status') == 'running':
                    # 保存停止状态
                    save_status("stopped", "用户手动停止")
                    print("停止信号已发送，任务将在当前目录处理完成后停止")
                    return 0
                else:
                    print(f"任务状态为: {status_data.get('status')}, 无需停止")
                    return 0
            else:
                print("未找到运行中的任务")
                return 1
        except Exception as e:
            print(f"停止任务失败: {e}")
            return 1

    # 后台运行模式
    if args.daemon:
        # 检查是否已有任务在运行
        if STATUS_FILE.exists():
            try:
                with open(STATUS_FILE, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
                if status_data.get('status') == 'running':
                    print(f"已有任务在运行中，开始时间: {status_data.get('start_time')}")
                    print("请先停止现有任务或使用 --status 查看状态")
                    return 1
            except:
                pass
        
        # 启动后台进程
        daemon_log_file = LOG_DIR / f"daemon_{datetime.now().strftime('%Y-%m-%d_%H%M')}.log"
        
        # 使用nohup启动后台进程
        cmd = [
            sys.executable, __file__,
            "--base-dir", args.base_dir,
            "--continue-on-error" if args.continue_on_error else "--no-continue-on-error"
        ]
        if args.filter:
            cmd.extend(["--filter", args.filter])
        
        # 重定向输出到日志文件
        with open(daemon_log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent,
                start_new_session=True  # 创建新的会话，脱离终端
            )
        
        print(f"后台任务已启动")
        print(f"进程ID: {process.pid}")
        print(f"日志文件: {daemon_log_file}")
        print(f"使用 --status 查看运行状态")
        print(f"使用 --stop 停止任务")
        return 0

    # 设置日志
    main_log_file = LOG_DIR / f"auto_batch_{datetime.now().strftime('%Y-%m-%d_%H%M')}.log"
    logger = setup_logging(main_log_file)
    
    # 记录开始时间
    start_time = datetime.now()

    logger.info("=" * 60)
    logger.info("开始自动批量处理")
    logger.info(f"基础目录: {args.base_dir}")
    logger.info(f"主日志文件: {main_log_file}")
    if args.dry_run:
        logger.info("模式: DRY RUN (预览)")
    if args.filter:
        logger.info(f"过滤条件: {args.filter}")
    logger.info("=" * 60)

    try:
        # 查找所有目录
        directories = find_date_directories(args.base_dir)

        # 应用过滤器
        if args.filter:
            directories = [d for d in directories if args.filter in d.name]
            logger.info(f"过滤后剩余 {len(directories)} 个目录")

        if not directories:
            logger.warning("未找到任何目录")
            return 0

        logger.info(f"找到 {len(directories)} 个目录待处理:")
        for i, d in enumerate(directories, 1):
            logger.info(f"  [{i}] {d.name}")
        logger.info("")

        # 保存初始状态
        save_status("running", "开始批量处理", "", 0, len(directories), 0)

        # 处理统计
        total = len(directories)
        success = 0
        failed = 0
        failed_dirs = []

        # 逐个处理
        for i, directory in enumerate(directories, 1):
            # 检查是否收到停止信号
            if STATUS_FILE.exists():
                try:
                    with open(STATUS_FILE, "r", encoding="utf-8") as f:
                        status_data = json.load(f)
                    if status_data.get('status') == 'stopped':
                        logger.info("收到停止信号，结束处理")
                        break
                except:
                    pass

            logger.info(f"\n进度: [{i}/{total}]")
            
            # 更新状态
            save_status("running", f"正在处理: {directory.name}", directory.name, i-1, total, failed)

            success_flag, result = process_directory(directory, logger, args.dry_run)

            if success_flag:
                success += 1
            else:
                failed += 1
                failed_dirs.append((directory.name, result))

                if not args.continue_on_error:
                    logger.error("遇到错误，停止处理")
                    break

        # 保存最终状态
        if failed == 0:
            save_status("completed", f"批量处理完成，成功处理 {success} 个目录", "", success, total, failed)
        else:
            save_status("completed", f"批量处理完成，成功 {success}，失败 {failed}", "", success, total, failed)

        # 输出总结
        logger.info("")
        logger.info("=" * 60)
        logger.info("批量处理完成")
        logger.info(f"总目录数: {total}")
        logger.info(f"成功: {success}")
        logger.info(f"失败: {failed}")

        if failed_dirs:
            logger.info("\n失败的目录:")
            for dir_name, error in failed_dirs:
                logger.info(f"  - {dir_name}: {error}")

        logger.info(f"\n主日志: {main_log_file}")
        logger.info("=" * 60)

        return failed

    except Exception as e:
        logger.error(f"程序异常: {str(e)}", exc_info=True)
        save_status("error", f"程序异常: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
