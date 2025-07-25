#!/bin/bash

echo "🐍 Python环境安装脚本"
echo "========================"

# 更新系统包
echo "📦 更新系统包..."
sudo apt update -y

# 安装Python3和pip3
echo "🔧 安装Python3和pip3..."
sudo apt install -y python3 python3-pip python3-venv python3-dev

# 验证安装
echo "✅ 验证Python安装..."
python3 --version
pip3 --version

# 升级pip
echo "⬆️ 升级pip..."
python3 -m pip install --upgrade pip

# 安装PyTorch（CPU版本，如果需要GPU版本请修改）
echo "🔥 安装PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
echo "📚 安装其他依赖..."
pip3 install transformers datasets accelerate

# 检查CUDA支持
echo "🎮 检查CUDA支持..."
python3 -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU数量:', torch.cuda.device_count())
    print('当前GPU:', torch.cuda.get_device_name(0))
else:
    print('CUDA不可用，使用CPU模式')
"

echo ""
echo "🎯 Python环境安装完成！"
echo ""
echo "📋 下一步操作："
echo "1. 进入项目目录: cd zhoudingnuo-model-trainer"
echo "2. 安装项目依赖: pip3 install -r requirements.txt"
echo "3. 开始训练: cd train_component && python3 model_expansion.py"
echo "" 