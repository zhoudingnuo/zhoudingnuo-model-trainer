# 🚀 vast.ai 云GPU部署指南

## 📋 快速开始

### 1. 注册和充值
- 访问 [vast.ai](https://vast.ai/)
- 注册账号并验证邮箱
- 充值（建议$10-20开始）

### 2. 选择GPU实例

**推荐配置：**
| GPU类型 | 内存 | 价格/小时 | 适用场景 |
|---------|------|-----------|----------|
| RTX 4090 | 24GB | $0.3-0.5 | 中等模型训练 |
| RTX 3090 | 24GB | $0.2-0.4 | 性价比之选 |
| A100 | 40GB | $0.8-1.2 | 大模型训练 |
| H100 | 80GB | $2-4 | 超大模型训练 |

### 3. 创建实例

1. **点击 "Create Instance"**
2. **选择GPU**：搜索 "RTX 4090" 或 "A100"
3. **选择镜像**：
   ```
   推荐镜像：
   - pytorch/pytorch:latest
   - nvidia/cuda:11.8-devel-ubuntu20.04
   ```
4. **配置实例**：
   - Disk Space: 100GB
   - Jupyter: ✅ 开启
   - SSH: ✅ 开启

### 4. 自动部署

**方法1：使用部署脚本**
```bash
# 连接到实例后运行
wget https://raw.githubusercontent.com/your-repo/setup_vast_ai.sh
chmod +x setup_vast_ai.sh
./setup_vast_ai.sh
```

**方法2：手动安装**
```bash
# 更新系统
apt update && apt upgrade -y

# 安装基础工具
apt install -y wget curl git htop nano vim unzip

# 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
```

### 5. 上传项目文件

**方法1：通过Jupyter上传**
- 打开Jupyter Notebook
- 在文件浏览器中上传项目文件

**方法2：通过Git克隆**
```bash
git clone https://github.com/your-repo/your-project.git
cd your-project
```

**方法3：通过SCP上传**
```bash
# 从本地上传
scp -r ./your-project root@your-instance-ip:/root/
```

### 6. 运行训练

```bash
# 进入项目目录
cd /root/your-project

# 运行模型扩展训练
python model_expansion.py

# 或者启动Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 🔧 环境配置

### 系统要求
- **操作系统**: Ubuntu 20.04/22.04
- **Python**: 3.8+
- **CUDA**: 11.8+
- **内存**: 32GB+
- **磁盘**: 100GB+

### 依赖包列表
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0
peft>=0.4.0
bitsandbytes>=0.41.0
tqdm>=4.65.0
numpy>=1.24.0
pandas>=2.0.0
```

## 💰 成本优化

### 省钱技巧
1. **选择便宜时段** - 晚上和周末价格较低
2. **使用spot实例** - 价格更便宜，但可能被中断
3. **及时停止实例** - 不用时立即停止，按秒计费
4. **批量训练** - 一次训练多个模型

### 成本估算
| GPU类型 | 价格/小时 | 训练时间 | 总成本 |
|---------|-----------|----------|--------|
| RTX 4090 | $0.4 | 2-4小时 | $0.8-1.6 |
| A100 | $1.0 | 1-2小时 | $1.0-2.0 |
| H100 | $3.0 | 0.5-1小时 | $1.5-3.0 |

## 📊 监控和管理

### 系统监控
```bash
# GPU使用情况
nvidia-smi

# 系统资源
htop

# 磁盘使用
df -h

# 网络速度
speedtest-cli
```

### 训练监控
```bash
# 查看训练日志
tail -f training.log

# 监控GPU内存
watch -n 1 nvidia-smi

# 查看进程
ps aux | grep python
```

## 🔒 安全注意事项

1. **不要上传敏感数据**
2. **使用强密码**
3. **及时删除不需要的实例**
4. **监控费用使用情况**
5. **定期备份重要数据**

## 🛠️ 故障排除

### 常见问题

**连接问题**
```bash
# 检查网络连接
ping google.com

# 重启网络服务
systemctl restart networking
```

**GPU问题**
```bash
# 检查GPU状态
nvidia-smi

# 重启GPU服务
sudo systemctl restart nvidia-persistenced
```

**内存不足**
```bash
# 清理内存
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# 减少batch_size
# 在model_expansion.py中修改batch_size参数
```

**磁盘空间不足**
```bash
# 清理临时文件
rm -rf /tmp/*
rm -rf ~/.cache/pip

# 清理旧模型
find . -name "*.safetensors" -size +1G -delete
```

## 📁 文件结构

```
/root/your-project/
├── model_expansion.py      # 主训练脚本
├── model_distillation.py   # 模型蒸馏脚本
├── requirements.txt        # 依赖列表
├── setup_vast_ai.sh       # 部署脚本
├── data/                  # 训练数据
│   ├── training_data.jsonl
│   └── fixed_training_data.jsonl
├── expanded_models/       # 扩展后的模型
└── logs/                 # 训练日志
```

## 🎯 最佳实践

1. **预测试** - 在小实例上测试代码
2. **备份** - 定期备份训练好的模型
3. **监控** - 实时监控训练进度和资源使用
4. **优化** - 根据GPU大小调整训练参数
5. **清理** - 及时清理不需要的文件

## 📞 技术支持

- **vast.ai文档**: https://vast.ai/docs/
- **PyTorch文档**: https://pytorch.org/docs/
- **Transformers文档**: https://huggingface.co/docs/transformers/

---

**💡 提示**: 记得及时停止实例以节省费用！ 