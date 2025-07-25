# 模型训练组件 (Train Component)

这个目录包含了所有与模型训练相关的代码和资源。

## 目录结构

```
train_component/
├── data/                           # 训练数据目录
│   ├── batch_training_data.jsonl   # 批量训练数据
│   └── fixed_training_data.jsonl   # 修复后的训练数据
├── model_expansion.py              # 模型扩展训练脚本
├── model_distillation.py           # 模型蒸馏脚本
├── check_training.py               # 训练检查工具
├── fix_data.py                     # 数据修复工具
├── model_info.py                   # 模型信息查看工具
├── debug_model_expansion.bat       # 调试模式启动脚本
├── start_model_expansion.bat       # 正常模式启动脚本
├── temp_input.txt                  # 临时输入文件（调试用）
└── README.md                       # 本文件
```

## 主要功能

### 1. 模型扩展训练 (`model_expansion.py`)
- 支持多种预训练模型的扩展训练
- 可配置训练参数（epochs、batch size等）
- 支持梯度累积策略
- 自动数据加载和验证

### 2. 模型蒸馏 (`model_distillation.py`)
- 将大模型知识传递给小模型
- 支持多种蒸馏策略
- 可配置蒸馏参数

### 3. 模型管理工具
- **模型下载器** (`model_downloader.py`): 从Hugging Face下载用户选择的模型
- **模型对话器** (`model_chat.py`): 选择本地模型进行对话交互
- **模型信息** (`model_info.py`): 查看模型详细信息

### 4. 数据处理工具
- **数据修复** (`fix_data.py`): 修复训练数据格式问题
- **训练检查** (`check_training.py`): 检查训练环境和数据

### 5. 启动脚本
- **模型工具启动器** (`start_model_tools.bat/.sh`): 统一启动所有模型工具
- **正常模式** (`start_model_expansion.bat`): 启动模型扩展训练
- **调试模式** (`debug_model_expansion.bat`): 使用预设参数进行调试训练

## 使用方法

### 快速开始
1. **模型下载**: 运行 `start_model_tools.bat` 或 `./start_model_tools.sh`，选择"模型下载器"
2. **模型对话**: 运行启动器，选择"模型对话器"与下载的模型对话
3. **模型训练**: 运行启动器，选择"模型扩展训练"开始训练
4. **数据准备**: 确保训练数据在 `data/` 目录中

### 数据准备
- 训练数据应为 JSONL 格式
- 每行包含一个 JSON 对象，格式为 `{"text": "训练文本内容"}`
- 使用 `fix_data.py` 修复数据格式问题

### 训练配置
- 修改脚本中的参数来调整训练设置
- 支持命令行参数配置
- 可自定义模型目录和数据目录

## 注意事项

1. 确保有足够的磁盘空间存储模型和训练数据
2. 训练过程可能需要较长时间，建议使用GPU加速
3. 定期备份重要的训练数据和模型文件
4. 在开始大规模训练前，建议先用小数据集测试

## 依赖项

请确保安装了 `requirements.txt` 中列出的所有依赖包。 