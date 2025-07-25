#!/bin/bash

echo "🚀 vast.ai 环境一键安装脚本"
echo "================================"

# 更新系统包
echo "📦 更新系统包..."
sudo apt update -y

# 安装基本工具
echo "🔧 安装基本工具..."
sudo apt install -y \
    iputils-ping \
    curl \
    wget \
    git \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    software-properties-common

# 检查GPU
echo "🎮 检查GPU环境..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 驱动已安装"
    nvidia-smi
else
    echo "❌ NVIDIA GPU 驱动未安装"
fi

# 检查CUDA
echo "🔬 检查CUDA环境..."
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA 已安装"
    nvcc --version
else
    echo "❌ CUDA 未安装"
fi

# 测试网络连接
echo "🌐 测试网络连接..."
if ping -c 1 github.com &> /dev/null; then
    echo "✅ 网络连接正常"
else
    echo "❌ 网络连接异常"
fi

# 测试HTTPS连接
echo "🔒 测试HTTPS连接..."
if curl -I https://github.com &> /dev/null; then
    echo "✅ HTTPS连接正常"
else
    echo "❌ HTTPS连接异常"
fi

echo ""
echo "🎯 安装完成！"
echo ""
echo "📋 下一步操作："
echo "1. 克隆项目: git clone https://github.com/zhoudingnuo/zhoudingnuo-model-trainer.git"
echo "2. 进入项目: cd zhoudingnuo-model-trainer"
echo "3. 安装依赖: pip3 install -r requirements.txt"
echo "4. 开始训练: cd train_component && python3 model_expansion.py"
echo "" 