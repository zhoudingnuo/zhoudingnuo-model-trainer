#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练状态实时监控脚本 - Ubuntu服务器优化版
支持持续监控GPU、进程、网络、磁盘等状态
"""

import torch
import os
import time
import psutil
import subprocess
import threading
from datetime import datetime
import signal
import sys

class TrainingMonitor:
    def __init__(self, refresh_interval=5):
        """
        初始化监控器
        
        Args:
            refresh_interval: 刷新间隔（秒）
        """
        self.refresh_interval = refresh_interval
        self.running = True
        self.start_time = time.time()
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """信号处理器，优雅退出"""
        print(f"\n🛑 收到退出信号 {signum}，正在停止监控...")
        self.running = False
    
    def get_gpu_info(self):
        """获取GPU信息"""
        gpu_info = {}
        
        if torch.cuda.is_available():
            try:
                # 基础GPU信息
                device = torch.cuda.current_device()
                gpu_info['name'] = torch.cuda.get_device_name(device)
                gpu_info['total_memory'] = torch.cuda.get_device_properties(device).total_memory / 1024**3
                gpu_info['allocated_memory'] = torch.cuda.memory_allocated(device) / 1024**3
                gpu_info['reserved_memory'] = torch.cuda.memory_reserved(device) / 1024**3
                gpu_info['free_memory'] = gpu_info['total_memory'] - gpu_info['allocated_memory']
                gpu_info['memory_usage_percent'] = (gpu_info['allocated_memory'] / gpu_info['total_memory']) * 100
                
                # 尝试获取GPU利用率
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info['gpu_utilization'] = utilization.gpu
                    gpu_info['memory_utilization'] = utilization.memory
                    
                    # 获取GPU温度
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        gpu_info['temperature'] = temp
                    except:
                        gpu_info['temperature'] = None
                        
                except Exception as e:
                    gpu_info['gpu_utilization'] = None
                    gpu_info['memory_utilization'] = None
                    gpu_info['temperature'] = None
                    
            except Exception as e:
                gpu_info['error'] = str(e)
        else:
            gpu_info['error'] = "CUDA不可用"
            
        return gpu_info
    
    def get_system_info(self):
        """获取系统信息"""
        try:
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 内存信息
            memory = psutil.virtual_memory()
            
            # 磁盘信息
            disk = psutil.disk_usage('/')
            
            # 网络信息
            network = psutil.net_io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_total': memory.total / 1024**3,
                'memory_used': memory.used / 1024**3,
                'memory_percent': memory.percent,
                'disk_total': disk.total / 1024**3,
                'disk_used': disk.used / 1024**3,
                'disk_percent': (disk.used / disk.total) * 100,
                'network_bytes_sent': network.bytes_sent / 1024**2,
                'network_bytes_recv': network.bytes_recv / 1024**2
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_training_processes(self):
        """获取训练相关进程"""
        training_processes = []
        
        try:
            # 查找Python训练进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                try:
                    if proc.info['name'] == 'python' or proc.info['name'] == 'python3':
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        if any(keyword in cmdline.lower() for keyword in ['train', 'expansion', 'model', 'pytorch']):
                            training_processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                                'cpu_percent': proc.info['cpu_percent'],
                                'memory_mb': proc.info['memory_info'].rss / 1024**2 if proc.info['memory_info'] else 0
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            training_processes.append({'error': str(e)})
            
        return training_processes
    
    def get_output_dirs(self):
        """检查输出目录"""
        output_dirs = []
        
        try:
            # 检查当前目录
            for item in os.listdir('.'):
                if item.startswith(('expanded_model_', 'output_', 'checkpoint_', 'trained_')):
                    item_path = os.path.join('.', item)
                    if os.path.isdir(item_path):
                        # 获取目录大小
                        total_size = 0
                        file_count = 0
                        for root, dirs, files in os.walk(item_path):
                            for file in files:
                                try:
                                    file_path = os.path.join(root, file)
                                    total_size += os.path.getsize(file_path)
                                    file_count += 1
                                except:
                                    pass
                        
                        # 检查是否有checkpoint
                        checkpoints = []
                        for root, dirs, files in os.walk(item_path):
                            for dir_name in dirs:
                                if 'checkpoint' in dir_name:
                                    checkpoints.append(dir_name)
                        
                        output_dirs.append({
                            'name': item,
                            'size_gb': total_size / 1024**3,
                            'file_count': file_count,
                            'checkpoints': checkpoints[:5]  # 只显示前5个
                        })
                        
        except Exception as e:
            output_dirs.append({'error': str(e)})
            
        return output_dirs
    
    def clear_screen(self):
        """清屏"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        """打印头部信息"""
        uptime = time.time() - self.start_time
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        
        print("=" * 80)
        print(f"🤖 训练状态实时监控器 - Ubuntu服务器版")
        print(f"⏰ 运行时间: {hours:02d}:{minutes:02d}:{seconds:02d} | 🔄 刷新间隔: {self.refresh_interval}s")
        print(f"📅 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def print_gpu_status(self, gpu_info):
        """打印GPU状态"""
        print("🎮 GPU状态:")
        print("-" * 40)
        
        if 'error' in gpu_info:
            print(f"❌ {gpu_info['error']}")
            return
        
        print(f"📱 设备名称: {gpu_info['name']}")
        print(f"💾 总内存: {gpu_info['total_memory']:.2f} GB")
        print(f"📊 已用内存: {gpu_info['allocated_memory']:.2f} GB ({gpu_info['memory_usage_percent']:.1f}%)")
        print(f"🗄️  缓存内存: {gpu_info['reserved_memory']:.2f} GB")
        print(f"🆓 可用内存: {gpu_info['free_memory']:.2f} GB")
        
        if gpu_info['gpu_utilization'] is not None:
            print(f"⚡ GPU利用率: {gpu_info['gpu_utilization']}%")
        if gpu_info['memory_utilization'] is not None:
            print(f"💾 显存利用率: {gpu_info['memory_utilization']}%")
        if gpu_info['temperature'] is not None:
            temp = gpu_info['temperature']
            temp_icon = "🔥" if temp > 80 else "🌡️" if temp > 60 else "❄️"
            print(f"{temp_icon} GPU温度: {temp}°C")
    
    def print_system_status(self, sys_info):
        """打印系统状态"""
        print("\n🖥️  系统状态:")
        print("-" * 40)
        
        if 'error' in sys_info:
            print(f"❌ {sys_info['error']}")
            return
        
        print(f"🖥️  CPU使用率: {sys_info['cpu_percent']:.1f}% ({sys_info['cpu_count']} 核心)")
        print(f"💾 内存使用: {sys_info['memory_used']:.1f}GB / {sys_info['memory_total']:.1f}GB ({sys_info['memory_percent']:.1f}%)")
        print(f"💿 磁盘使用: {sys_info['disk_used']:.1f}GB / {sys_info['disk_total']:.1f}GB ({sys_info['disk_percent']:.1f}%)")
        print(f"🌐 网络发送: {sys_info['network_bytes_sent']:.1f}MB | 接收: {sys_info['network_bytes_recv']:.1f}MB")
    
    def print_training_processes(self, processes):
        """打印训练进程"""
        print("\n🚀 训练进程:")
        print("-" * 40)
        
        if not processes:
            print("📭 未发现训练进程")
            return
        
        for proc in processes:
            if 'error' in proc:
                print(f"❌ {proc['error']}")
                continue
                
            print(f"🔄 PID {proc['pid']}: {proc['name']}")
            print(f"   📝 命令: {proc['cmdline']}")
            print(f"   🖥️  CPU: {proc['cpu_percent']:.1f}% | 💾 内存: {proc['memory_mb']:.1f}MB")
            print()
    
    def print_output_dirs(self, dirs):
        """打印输出目录"""
        print("📁 输出目录:")
        print("-" * 40)
        
        if not dirs:
            print("📭 未发现输出目录")
            return
        
        for dir_info in dirs:
            if 'error' in dir_info:
                print(f"❌ {dir_info['error']}")
                continue
                
            print(f"📂 {dir_info['name']}")
            print(f"   📊 大小: {dir_info['size_gb']:.2f}GB | 📄 文件数: {dir_info['file_count']}")
            if dir_info['checkpoints']:
                print(f"   💾 Checkpoints: {', '.join(dir_info['checkpoints'])}")
            print()
    
    def run_monitor(self):
        """运行监控器"""
        print("🚀 启动训练状态监控器...")
        print("💡 按 Ctrl+C 退出监控")
        print()
        
        while self.running:
            try:
                # 清屏
                self.clear_screen()
                
                # 打印头部
                self.print_header()
                
                # 获取并打印各种状态
                gpu_info = self.get_gpu_info()
                self.print_gpu_status(gpu_info)
                
                sys_info = self.get_system_info()
                self.print_system_status(sys_info)
                
                processes = self.get_training_processes()
                self.print_training_processes(processes)
                
                dirs = self.get_output_dirs()
                self.print_output_dirs(dirs)
                
                # 等待下次刷新
                time.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 用户中断，正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 监控出错: {e}")
                time.sleep(self.refresh_interval)
        
        print("👋 监控器已停止")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练状态实时监控器')
    parser.add_argument('--interval', '-i', type=int, default=5, 
                       help='刷新间隔（秒，默认5秒）')
    parser.add_argument('--no-clear', action='store_true',
                       help='不清屏（用于日志记录）')
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = TrainingMonitor(refresh_interval=args.interval)
    
    # 运行监控
    monitor.run_monitor()

if __name__ == "__main__":
    main() 