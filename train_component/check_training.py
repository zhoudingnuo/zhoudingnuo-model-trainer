#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查训练状态的脚本
"""

import torch
import os
import time

def check_gpu_status():
    """检查GPU状态"""
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"GPU已用内存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU缓存内存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU利用率: {utilization.gpu}%")
        except Exception as e:
            print(f"无法获取GPU利用率: {e}")
    else:
        print("CUDA不可用")

def check_output_dirs():
    """检查输出目录"""
    base_dir = "."
    for item in os.listdir(base_dir):
        if item.startswith("expanded_model_"):
            print(f"找到输出目录: {item}")
            # 检查是否有checkpoint
            checkpoint_dir = os.path.join(item, "checkpoint-1000")
            if os.path.exists(checkpoint_dir):
                print(f"  - 找到checkpoint: {checkpoint_dir}")
            else:
                print(f"  - 未找到checkpoint")

if __name__ == "__main__":
    print("=== 训练状态检查 ===")
    check_gpu_status()
    print("\n=== 输出目录检查 ===")
    check_output_dirs() 