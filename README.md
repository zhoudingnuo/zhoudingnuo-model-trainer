# 模型扩展训练项目

## 项目简介

这是一个用于模型扩展和训练的完整解决方案，支持：
- 读取本地模型文件夹 `D:\Model` 中的模型，让用户选择要扩展的模型
- 实现模型的覆盖式增训，支持从1.3b扩展到3b、7b、13b等更大模型
- 使用训练数据对扩展后的模型进行训练
- **智能梯度累积**：自动根据GPU内存调整批次大小和梯度累积步数，实现大批次训练效果
- **云GPU支持**：支持vast.ai等云GPU平台部署

### 2. 模型蒸馏 (`model_distillation.py`)
- 读取本地模型文件夹 `D:\Model` 中的模型，让用户选择要蒸馏的模型
- 实现知识蒸馏，创建更小、更高效的模型
- 支持多种蒸馏策略和损失函数

## 🚀 快速开始

### 本地运行

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **准备数据**
- 将训练数据放在 `data/` 文件夹中
- 支持 `.jsonl` 格式

3. **运行训练**
```bash
cd train_component
python model_expansion.py
```

### 云GPU部署 (vast.ai)

1. **创建vast.ai实例**
- 选择RTX 4090或A100 GPU
- 使用预装PyTorch镜像

2. **自动部署**
```bash
# 上传项目文件后运行
chmod +x scripts/setup_vast_ai.sh
./scripts/setup_vast_ai.sh
```

3. **开始训练**
```bash
cd train_component
python model_expansion.py
```

详细部署指南请参考：[VAST_AI_GUIDE.md](VAST_AI_GUIDE.md)

## 📁 项目结构

```
zhoudingnuo-model-trainer/
├── train_component/            # 🎯 模型训练组件目录
│   ├── model_expansion.py      # 模型扩展训练主程序
│   ├── model_distillation.py   # 模型蒸馏程序
│   ├── check_training.py       # 训练检查工具
│   ├── fix_data.py             # 数据修复工具
│   ├── model_info.py           # 模型信息查看工具
│   ├── debug_model_expansion.bat # Windows调试脚本
│   ├── start_model_expansion.bat # Windows启动脚本
│   ├── data/                   # 训练数据目录
│   │   ├── batch_training_data.jsonl
│   │   └── fixed_training_data.jsonl
│   └── README.md               # 训练组件说明文档
├── scripts/                    # 🔧 脚本工具集
│   ├── setup_vast_ai.sh        # vast.ai自动部署脚本
│   ├── setup_github.bat        # GitHub仓库初始化(Windows)
│   ├── upload_to_github.bat    # 上传代码到GitHub(Windows)
│   ├── update_from_github.bat  # 从GitHub更新代码(Windows)
│   ├── upload_to_github.sh     # 上传代码到GitHub(Linux/Mac)
│   ├── update_from_github.sh   # 从GitHub更新代码(Linux/Mac)
│   ├── GITHUB_GUIDE.md         # GitHub使用详细指南
│   ├── VAST_AI_GUIDE.md        # vast.ai使用详细指南
│   └── README.md               # 脚本工具说明文档
├── requirements.txt            # Python依赖包列表
└── expanded_models/            # 扩展后的模型保存目录
```

## 🔧 环境要求

### 本地环境
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU训练)
- 至少8GB GPU内存

### 云GPU环境
- Ubuntu 20.04/22.04
- CUDA 11.8+
- 32GB+ 系统内存
- 100GB+ 磁盘空间

## 📦 依赖包

### 核心依赖
```
torch>=2.0.0                    # PyTorch深度学习框架
transformers>=4.30.0            # Hugging Face Transformers
datasets>=2.12.0                # 数据集处理
accelerate>=0.20.0              # 分布式训练加速
```

### 训练优化
```
peft>=0.4.0                     # 参数高效微调
bitsandbytes>=0.41.0            # 量化训练
tqdm>=4.65.0                    # 进度条
```

### 数据处理
```
numpy>=1.24.0                   # 数值计算
pandas>=2.0.0                   # 数据处理
scikit-learn>=1.3.0             # 机器学习工具
```

### 可视化
```
matplotlib>=3.7.0               # 图表绘制
seaborn>=0.12.0                 # 统计图表
```

## 🎯 功能特性

### 模型扩展
- **智能层复制**：保持原有知识的同时扩展模型容量
- **权重初始化**：新层采用合理的初始化策略
- **多尺寸支持**：支持1b、1.8b、3b、7b、13b等不同尺寸

### 训练优化
- **梯度累积**：使用小批次模拟大批次训练效果
- **混合精度**：FP16训练减少内存使用
- **自适应参数**：根据GPU内存自动调整训练参数
- **内存监控**：实时监控GPU内存使用情况

### 云GPU支持
- **一键部署**：自动化环境配置
- **成本优化**：智能选择性价比最高的GPU
- **监控管理**：实时监控训练进度和资源使用

## 💰 成本估算

### vast.ai价格参考
| GPU类型 | 内存 | 价格/小时 | 训练时间 | 总成本 |
|---------|------|-----------|----------|--------|
| RTX 4090 | 24GB | $0.4 | 2-4小时 | $0.8-1.6 |
| A100 | 40GB | $1.0 | 1-2小时 | $1.0-2.0 |
| H100 | 80GB | $3.0 | 0.5-1小时 | $1.5-3.0 |

## 📊 使用示例

### 1. 模型扩展训练
```bash
# 运行模型扩展训练
python model_expansion.py

# 选择模型和参数
# 1. 选择要扩展的模型
# 2. 选择目标大小 (1b, 1.8b, 3b, 7b)
# 3. 设置训练参数 (epochs, batch_size)
# 4. 开始训练
```

### 2. 模型蒸馏
```bash
# 运行模型蒸馏
python model_distillation.py

# 选择要蒸馏的模型
# 设置蒸馏参数
# 开始蒸馏
```

### 3. 云GPU训练
```bash
# 在vast.ai实例上
./setup_vast_ai.sh              # 自动部署环境
python model_expansion.py       # 开始训练
```

### 4. GitHub代码管理
```bash
# 初始化GitHub仓库
setup_github.bat                # Windows
./setup_github.sh              # Linux/Mac

# 上传代码到GitHub
upload_to_github.bat "提交信息"  # Windows
./upload_to_github.sh "提交信息" # Linux/Mac

# 从GitHub更新代码
update_from_github.bat          # Windows
./update_from_github.sh         # Linux/Mac
```

## 🔍 监控和调试

### 本地监控
```bash
# 查看GPU使用情况
nvidia-smi

# 监控训练进度
tail -f training.log

# 检查模型文件
ls -lh expanded_models/
```

### 云GPU监控
```bash
# 系统资源监控
htop

# GPU监控
watch -n 1 nvidia-smi

# 磁盘使用
df -h
```

## 🛠️ 故障排除

### 常见问题

**内存不足**
- 减少batch_size
- 启用梯度检查点
- 使用混合精度训练

**训练速度慢**
- 检查GPU使用率
- 优化数据加载
- 使用更快的GPU

**依赖安装失败**
- 更新pip: `pip install --upgrade pip`
- 使用conda环境
- 检查Python版本兼容性

## 📈 性能优化

### GPU优化
- **批次大小**：根据GPU内存调整
- **序列长度**：平衡训练效果和内存使用
- **梯度累积**：模拟大批次训练

### 训练优化
- **学习率调度**：使用warmup和衰减
- **权重衰减**：防止过拟合
- **早停机制**：避免过度训练

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 支持

- **问题反馈**：创建 GitHub Issue
- **功能建议**：提交 Feature Request
- **技术交流**：加入讨论区

---

**🎯 提示**：
- 首次使用建议在小数据集上测试
- 云GPU训练记得及时停止实例节省费用
- 定期备份训练好的模型 