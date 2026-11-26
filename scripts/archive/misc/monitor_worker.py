#!/usr/bin/env python3
"""
Huey Worker 日志监控脚本

功能:
1. 实时监控 huey_worker.log
2. 统计成功/失败任务数
3. 检测ERROR关键字并报警
4. 每10秒输出进度摘要
5. 支持Ctrl+C优雅退出

使用方法:
    python scripts/monitor_worker.py
    python scripts/monitor_worker.py --log-file huey_worker.log
    python scripts/monitor_worker.py --interval 5
"""

import os
import sys
import time
import signal
import argparse
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict


class WorkerMonitor:
    """Huey Worker 日志监控器"""

    def __init__(self, log_file: str, interval: int = 10):
        self.log_file = Path(log_file)
        self.interval = interval
        self.running = True

        # 统计计数器
        self.stats = {
            'success': 0,
            'failed': 0,
            'errors': 0,
            'warnings': 0,
            'total_lines': 0,
        }

        # 任务统计
        self.task_stats = defaultdict(lambda: {'success': 0, 'failed': 0})

        # 最近的错误信息
        self.recent_errors = []
        self.max_recent_errors = 10

        # 文件位置
        self.file_position = 0

        # 上次摘要时间
        self.last_summary_time = time.time()

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        print("\n")
        self._print_separator()
        print("[INFO] 收到退出信号，正在优雅退出...")
        self.running = False

    def _print_separator(self):
        """打印分隔线"""
        print("=" * 60)

    def _print_header(self):
        """打印头部信息"""
        self._print_separator()
        print(f"  Huey Worker 日志监控器")
        print(f"  日志文件: {self.log_file}")
        print(f"  摘要间隔: {self.interval}秒")
        print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._print_separator()
        print("[INFO] 按 Ctrl+C 优雅退出")
        print()

    def _parse_line(self, line: str):
        """解析日志行"""
        self.stats['total_lines'] += 1
        line_lower = line.lower()

        # 检测错误
        if 'error' in line_lower or 'exception' in line_lower:
            self.stats['errors'] += 1
            self._alert_error(line)
            return

        # 检测警告
        if 'warning' in line_lower or 'warn' in line_lower:
            self.stats['warnings'] += 1
            return

        # 检测任务成功 (Huey 格式)
        # 例如: Executing crawl_task
        # 例如: crawl_task executed in 1.23s
        success_patterns = [
            r'executed in [\d.]+s',
            r'task completed',
            r'success',
        ]
        for pattern in success_patterns:
            if re.search(pattern, line_lower):
                self.stats['success'] += 1
                # 尝试提取任务名
                task_match = re.search(r'(\w+_task)', line)
                if task_match:
                    task_name = task_match.group(1)
                    self.task_stats[task_name]['success'] += 1
                return

        # 检测任务失败
        fail_patterns = [
            r'task failed',
            r'failed to execute',
            r'retry',
        ]
        for pattern in fail_patterns:
            if re.search(pattern, line_lower):
                self.stats['failed'] += 1
                task_match = re.search(r'(\w+_task)', line)
                if task_match:
                    task_name = task_match.group(1)
                    self.task_stats[task_name]['failed'] += 1
                return

    def _alert_error(self, line: str):
        """错误报警"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        error_msg = line.strip()[:100]  # 截断过长的消息

        # 保存最近错误
        self.recent_errors.append({
            'time': timestamp,
            'message': error_msg
        })
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)

        # 打印报警
        print(f"\n[ALERT] [{timestamp}] 检测到错误!")
        print(f"        {error_msg}")
        print()

    def _print_summary(self):
        """打印进度摘要"""
        now = datetime.now().strftime('%H:%M:%S')

        print()
        self._print_separator()
        print(f"  进度摘要 [{now}]")
        self._print_separator()
        print(f"  总行数:     {self.stats['total_lines']}")
        print(f"  成功任务:   {self.stats['success']}")
        print(f"  失败任务:   {self.stats['failed']}")
        print(f"  错误数:     {self.stats['errors']}")
        print(f"  警告数:     {self.stats['warnings']}")

        # 任务统计
        if self.task_stats:
            print()
            print("  任务统计:")
            for task_name, stats in self.task_stats.items():
                print(f"    {task_name}: 成功={stats['success']}, 失败={stats['failed']}")

        # 最近错误
        if self.recent_errors:
            print()
            print(f"  最近错误 (最多{self.max_recent_errors}条):")
            for err in self.recent_errors[-5:]:  # 只显示最近5条
                print(f"    [{err['time']}] {err['message'][:50]}...")

        self._print_separator()
        print()

    def _read_new_lines(self):
        """读取新的日志行"""
        if not self.log_file.exists():
            return []

        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                # 移动到上次读取的位置
                f.seek(self.file_position)
                lines = f.readlines()
                self.file_position = f.tell()
                return lines
        except Exception as e:
            print(f"[WARN] 读取日志文件失败: {e}")
            return []

    def run(self):
        """运行监控"""
        self._print_header()

        # 检查日志文件
        if not self.log_file.exists():
            print(f"[WARN] 日志文件不存在: {self.log_file}")
            print("[INFO] 等待日志文件创建...")

        # 从文件末尾开始读取
        if self.log_file.exists():
            self.file_position = self.log_file.stat().st_size
            print(f"[INFO] 从文件末尾开始监控 (位置: {self.file_position})")

        while self.running:
            try:
                # 读取新行
                new_lines = self._read_new_lines()
                for line in new_lines:
                    line = line.strip()
                    if line:
                        self._parse_line(line)
                        # 实时输出日志行
                        print(f"[LOG] {line[:120]}")

                # 检查是否需要输出摘要
                current_time = time.time()
                if current_time - self.last_summary_time >= self.interval:
                    self._print_summary()
                    self.last_summary_time = current_time

                # 短暂休眠
                time.sleep(0.5)

            except Exception as e:
                print(f"[ERROR] 监控异常: {e}")
                time.sleep(1)

        # 退出时打印最终摘要
        print()
        print("[INFO] 最终统计:")
        self._print_summary()
        print("[INFO] 监控已停止")


def main():
    parser = argparse.ArgumentParser(description='Huey Worker 日志监控')
    parser.add_argument(
        '--log-file', '-f',
        default='E:/JETB/pythons/sf/retrival/huey_worker.log',
        help='日志文件路径'
    )
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=10,
        help='摘要输出间隔(秒)'
    )

    args = parser.parse_args()

    monitor = WorkerMonitor(
        log_file=args.log_file,
        interval=args.interval
    )
    monitor.run()


if __name__ == '__main__':
    main()
