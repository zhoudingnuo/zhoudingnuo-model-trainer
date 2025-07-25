# vast.ai 云GPU部署详细指南

## 🎯 快速开始

### 1. 连接到vast.ai实例

```bash
ssh root@[vast.ai实例IP地址]
```

### 2. 一键部署

```bash
# 克隆项目
git clone https://github.com/zhoudingnuo/zhoudingnuo-model-trainer.git
cd zhoudingnuo-model-trainer

# 运行自动部署脚本
chmod +x scripts/setup_vast_ai.sh
./scripts/setup_vast_ai.sh

# 进入训练目录
cd train_component

# 开始训练
python model_expansion.py
```

## 🔧 环境检查

### 检查GPU状态
```bash
nvidia-smi
```

### 检查Python环境
```bash
python --version
pip list | grep torch
```

### 检查CUDA版本
```bash
nvcc --version
```

## 📁 项目结构

部署后的项目结构：
```
zhoudingnuo-model-trainer/
├── train_component/          # 训练组件
│   ├── model_expansion.py    # 模型扩展训练
│   ├── model_distillation.py # 模型蒸馏
│   ├── data/                 # 训练数据
│   └── ...
├── scripts/                  # 脚本工具
│   ├── setup_vast_ai.sh      # 自动部署脚本
│   └── ...
└── requirements.txt          # Python依赖
```

## 🚀 训练配置

### 推荐配置（140GB GPU）

```bash
# 进入训练目录
cd train_component

# 运行训练脚本
python model_expansion.py

# 选择配置：
# - 模型: 选择适合的预训练模型
# - 扩展大小: 根据GPU内存选择
# - 批次大小: 可以设置较大值（如16-32）
# - 训练轮数: 根据数据量调整
```

### 性能优化建议

1. **批次大小**: 140GB GPU可以支持很大的批次大小
2. **梯度累积**: 自动处理，无需手动设置
3. **混合精度**: 自动启用，节省显存
4. **数据并行**: 如果有多个GPU，可以启用

## 📊 监控训练

### 实时监控GPU使用
```bash
watch -n 1 nvidia-smi
```

### 查看训练日志
```bash
tail -f logs/training.log
```

### 检查训练进度
```bash
ls -la expanded_models/
```

## 💾 数据管理

### 上传训练数据
```bash
# 使用scp上传数据
scp -r /path/to/your/data root@[vast.ai-IP]:~/zhoudingnuo-model-trainer/train_component/data/
```

### 下载训练结果
```bash
# 下载训练好的模型
scp -r root@[vast.ai-IP]:~/zhoudingnuo-model-trainer/expanded_models/ /local/path/
```

## 🔄 持续训练

### 恢复训练
```bash
cd train_component
python model_expansion.py --resume_from_checkpoint /path/to/checkpoint
```

### 定期保存
```bash
# 训练脚本会自动保存检查点
# 检查点位置: expanded_models/checkpoints/
```

## ⚠️ 注意事项

1. **成本控制**: 140GB GPU成本较高，建议合理规划训练时间
2. **数据备份**: 重要数据要及时下载到本地
3. **实例管理**: 注意vast.ai实例的计费时间
4. **网络稳定**: 确保网络连接稳定，避免训练中断

## 🆘 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   # 或启用梯度检查点
   ```

2. **依赖包安装失败**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

3. **训练中断**
   ```bash
   # 检查是否有检查点文件
   ls -la expanded_models/checkpoints/
   ```

## 📞 技术支持

如果遇到问题，请检查：
1. vast.ai实例状态
2. GPU驱动版本
3. Python环境
4. 训练日志

---

**祝您训练顺利！** 🎉 