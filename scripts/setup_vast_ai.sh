#!/bin/bash

# vast.ai Auto Deployment Script
# For quick deployment of model expansion training environment on vast.ai instances

echo "🚀 Starting deployment of model expansion training environment..."

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install basic tools
echo "🔧 Installing basic tools..."
apt install -y wget curl git htop nano vim unzip

# Check CUDA version
echo "🔍 Checking CUDA environment..."
nvidia-smi
nvcc --version

# Install Python dependencies
echo "🐍 Installing Python dependencies..."

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (based on CUDA version)
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c1-4)
echo "Detected CUDA version: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == "11.8" ]]; then
    echo "Installing PyTorch for CUDA 11.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    echo "Installing PyTorch for CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install project dependencies
echo "📋 Installing project dependencies..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
"

# Install transformers
echo "🤗 Installing Transformers..."
pip install transformers datasets accelerate

# Create project directory
echo "📁 Creating project directory..."
mkdir -p /root/model_expansion_project
cd /root/model_expansion_project

# Download project files (if via git)
# git clone https://github.com/your-repo/your-project.git .

echo "🎯 Environment deployment completed!"
echo "📊 System information:"
echo "  - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "  - Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "  - Disk: $(df -h / | tail -1 | awk '{print $4}') available"

echo ""
echo "🚀 Next steps:"
echo "  1. Upload your project files to /root/model_expansion_project/"
echo "  2. Run: python model_expansion.py"
echo "  3. Or start Jupyter: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""
echo "💡 Tips:"
echo "  - Use 'nvidia-smi' to monitor GPU usage"
echo "  - Use 'htop' to monitor system resources"
echo "  - Remember to stop instances promptly to save costs" 