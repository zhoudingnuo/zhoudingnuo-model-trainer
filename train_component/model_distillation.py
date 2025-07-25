import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import json
import glob
from typing import List, Dict, Any
import argparse

class DistillationTrainer(Trainer):
    """
    自定义蒸馏训练器
    """
    def __init__(self, teacher_model, temperature=0.5, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算蒸馏损失
        """
        # 学生模型前向传播
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # 教师模型前向传播（不计算梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # 计算蒸馏损失
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 计算任务损失（如果有标签的话）
        task_loss = student_outputs.loss if hasattr(student_outputs, 'loss') else 0
        
        # 总损失
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss
        
        return (total_loss, student_outputs) if return_outputs else total_loss

class ModelDistiller:
    def __init__(self, model_dir: str = "D:\\Model", data_dir: str = "data"):
        """
        初始化模型蒸馏器
        
        Args:
            model_dir: 模型文件夹路径
            data_dir: 数据文件夹路径
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
    def list_models(self) -> List[str]:
        """
        列出模型文件夹中的所有模型
        
        Returns:
            模型路径列表
        """
        if not os.path.exists(self.model_dir):
            print(f"模型文件夹 {self.model_dir} 不存在")
            return []
            
        models = []
        for item in os.listdir(self.model_dir):
            item_path = os.path.join(self.model_dir, item)
            if os.path.isdir(item_path):
                # 检查是否包含模型文件
                try:
                    files = os.listdir(item_path)
                    
                    # 检查是否是Hugging Face Hub格式（包含snapshots文件夹）
                    if 'snapshots' in files:
                        # 检查snapshots文件夹中的内容
                        snapshots_path = os.path.join(item_path, 'snapshots')
                        if os.path.exists(snapshots_path):
                            snapshot_dirs = os.listdir(snapshots_path)
                            if snapshot_dirs:
                                # 检查第一个snapshot目录
                                first_snapshot = os.path.join(snapshots_path, snapshot_dirs[0])
                                if os.path.exists(first_snapshot):
                                    snapshot_files = os.listdir(first_snapshot)
                                    model_files = [f for f in snapshot_files if f.endswith(('.bin', '.safetensors', '.pt', '.pth'))]
                                    config_files = [f for f in snapshot_files if f in ('config.json', 'tokenizer.json', 'tokenizer_config.json')]
                                    
                                    if model_files or config_files:
                                        models.append(item)
                                        print(f"找到Hugging Face模型: {item}")
                                        continue
                    
                    # 检查常见的模型文件扩展名
                    model_files = [f for f in files if f.endswith(('.bin', '.safetensors', '.pt', '.pth'))]
                    # 或者检查是否包含配置文件
                    config_files = [f for f in files if f in ('config.json', 'tokenizer.json', 'tokenizer_config.json')]
                    
                    if model_files or config_files:
                        models.append(item)
                        print(f"找到模型: {item}")
                except Exception as e:
                    print(f"检查文件夹 {item} 时出错: {e}")
                    continue
                    
        return models
    
    def select_model(self) -> str:
        """
        让用户选择要蒸馏的模型
        
        Returns:
            选择的模型名称
        """
        models = self.list_models()
        
        if not models:
            print("未找到任何模型")
            return None
            
        print("可用的模型:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
            
        while True:
            try:
                choice = int(input(f"请选择要蒸馏的模型 (1-{len(models)}): ")) - 1
                if 0 <= choice < len(models):
                    selected_model = models[choice]
                    print(f"已选择模型: {selected_model}")
                    return selected_model
                else:
                    print("无效选择，请重试")
            except ValueError:
                print("请输入有效数字")
    
    def load_model_and_tokenizer(self, model_name: str):
        """
        加载模型和分词器
        
        Args:
            model_name: 模型名称
        """
        model_path = os.path.join(self.model_dir, model_name)
        
        try:
            print(f"正在加载模型: {model_path}")
            
            # 检查是否是Hugging Face Hub格式
            if os.path.exists(os.path.join(model_path, 'snapshots')):
                # 找到第一个snapshot目录
                snapshots_path = os.path.join(model_path, 'snapshots')
                snapshot_dirs = os.listdir(snapshots_path)
                if snapshot_dirs:
                    actual_model_path = os.path.join(snapshots_path, snapshot_dirs[0])
                    print(f"使用snapshot路径: {actual_model_path}")
                else:
                    actual_model_path = model_path
            else:
                actual_model_path = model_path
            
            self.tokenizer = AutoTokenizer.from_pretrained(actual_model_path)
            self.teacher_model = AutoModelForCausalLM.from_pretrained(actual_model_path)
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.teacher_model.to(self.device)
            print(f"教师模型加载成功，参数量: {self.teacher_model.num_parameters():,}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
            
        return True
    
    def load_training_data(self, max_lines: int = None) -> Dataset:
        """
        加载训练数据
        
        Args:
            max_lines: 最大加载行数，None表示加载全部
            
        Returns:
            训练数据集
        """
        if not os.path.exists(self.data_dir):
            print(f"数据文件夹 {self.data_dir} 不存在")
            return None
            
        data_files = []
        for ext in ['*.txt', '*.json', '*.jsonl']:
            data_files.extend(glob.glob(os.path.join(self.data_dir, ext)))
            
        if not data_files:
            print("未找到训练数据文件")
            return None
            
        print(f"找到数据文件: {data_files}")
        
        # 加载数据
        texts = []
        line_count = 0
        for file_path in data_files:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if max_lines and line_count >= max_lines:
                            break
                        if line.strip():
                            texts.append(line.strip())
                            line_count += 1
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if max_lines and line_count >= max_lines:
                                break
                            texts.append(str(item))
                            line_count += 1
                    else:
                        if not max_lines or line_count < max_lines:
                            texts.append(str(data))
                            line_count += 1
            elif file_path.endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if max_lines and line_count >= max_lines:
                            break
                        if line.strip():
                            try:
                                data = json.loads(line)
                                # 处理JSONL格式，提取text字段
                                if isinstance(data, dict) and 'text' in data:
                                    texts.append(data['text'])
                                else:
                                    texts.append(str(data))
                                line_count += 1
                            except json.JSONDecodeError:
                                print(f"跳过无效的JSON行: {line[:100]}...")
                                continue
                            
        if not texts:
            print("未找到有效训练数据")
            return None
            
        print(f"加载了 {len(texts)} 条训练数据")
        if max_lines:
            print(f"限制加载前 {max_lines} 行数据")
        
        # 创建数据集
        dataset = Dataset.from_dict({"text": texts})
        return dataset
    
    def tokenize_function(self, examples):
        """
        分词函数
        """
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def create_student_model(self, compression_ratio: float = 0.5):
        """
        创建学生模型（压缩后的模型）
        
        Args:
            compression_ratio: 压缩比例 (0-1)
        """
        print(f"创建学生模型，压缩比例: {compression_ratio}")
        
        # 获取教师模型配置
        teacher_config = self.teacher_model.config
        
        # 创建学生模型配置
        student_config = teacher_config.__class__.from_pretrained(
            os.path.join(self.model_dir, self.selected_model_name)
        )
        
        # 根据压缩比例调整配置
        student_config.hidden_size = max(512, int(teacher_config.hidden_size * compression_ratio))
        student_config.num_hidden_layers = max(6, int(teacher_config.num_hidden_layers * compression_ratio))
        student_config.num_attention_heads = max(8, int(teacher_config.num_attention_heads * compression_ratio))
        student_config.intermediate_size = student_config.hidden_size * 4
        
        # 创建学生模型
        self.student_model = AutoModelForCausalLM.from_config(student_config)
        self.student_model.to(self.device)
        
        print(f"学生模型创建完成，参数量: {self.student_model.num_parameters():,}")
        print(f"压缩比: {self.student_model.num_parameters() / self.teacher_model.num_parameters():.2%}")
        
        return True
    
    def create_custom_student_model(self, hidden_size: int, num_layers: int, num_heads: int):
        """
        创建自定义学生模型
        
        Args:
            hidden_size: 隐藏层大小
            num_layers: 层数
            num_heads: 注意力头数
        """
        print(f"创建自定义学生模型: hidden_size={hidden_size}, layers={num_layers}, heads={num_heads}")
        
        # 获取教师模型配置
        teacher_config = self.teacher_model.config
        
        # 创建学生模型配置
        student_config = teacher_config.__class__.from_pretrained(
            os.path.join(self.model_dir, self.selected_model_name)
        )
        
        # 设置自定义参数
        student_config.hidden_size = hidden_size
        student_config.num_hidden_layers = num_layers
        student_config.num_attention_heads = num_heads
        student_config.intermediate_size = hidden_size * 4
        
        # 创建学生模型
        self.student_model = AutoModelForCausalLM.from_config(student_config)
        self.student_model.to(self.device)
        
        print(f"自定义学生模型创建完成，参数量: {self.student_model.num_parameters():,}")
        print(f"压缩比: {self.student_model.num_parameters() / self.teacher_model.num_parameters():.2%}")
        
        return True
    
    def distill_model(self, output_dir: str, temperature: float = 0.5, alpha: float = 0.5, 
                     epochs: int = 3, batch_size: int = 4, max_lines: int = None):
        """
        蒸馏模型
        
        Args:
            output_dir: 输出目录
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            epochs: 训练轮数
            batch_size: 批次大小
            max_lines: 最大加载行数，None表示加载全部
        """
        # 加载训练数据
        dataset = self.load_training_data(max_lines)
        if dataset is None:
            return False
            
        # 分词
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=100,
            learning_rate=5e-5,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            remove_unused_columns=False,
        )
        
        # 创建蒸馏训练器
        trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            temperature=temperature,
            alpha=alpha,
            model=self.student_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )
        
        # 开始蒸馏训练
        print(f"开始蒸馏训练，温度: {temperature}, 权重: {alpha}")
        trainer.train()
        
        # 保存蒸馏后的模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"蒸馏完成，模型已保存到: {output_dir}")
        return True
    
    def run_distillation_pipeline(self):
        """
        运行完整的模型蒸馏流程
        """
        print("=== 模型蒸馏管道 ===")
        
        # 1. 选择模型
        self.selected_model_name = self.select_model()
        if self.selected_model_name is None:
            return
            
        # 2. 加载教师模型
        if not self.load_model_and_tokenizer(self.selected_model_name):
            return
            
        # 3. 选择压缩方式
        print("\n请选择压缩方式:")
        print("1. 使用压缩比例")
        print("2. 自定义参数")
        
        while True:
            try:
                choice = int(input("请选择 (1-2): "))
                if choice == 1:
                    # 使用压缩比例
                    while True:
                        try:
                            compression_ratio = float(input("请输入压缩比例 (0.1-0.9，默认0.5): ") or "0.5")
                            if 0.1 <= compression_ratio <= 0.9:
                                break
                            else:
                                print("压缩比例必须在0.1到0.9之间")
                        except ValueError:
                            print("请输入有效数字")
                    
                    # 4. 创建学生模型
                    if not self.create_student_model(compression_ratio):
                        return
                    break
                elif choice == 2:
                    # 自定义参数
                    print("\n请输入自定义参数:")
                    
                    while True:
                        try:
                            hidden_size = int(input(f"hidden_size (当前{self.teacher_model.config.hidden_size}，建议更小): ") or str(max(512, self.teacher_model.config.hidden_size // 2)))
                            if hidden_size > 0:
                                break
                            else:
                                print("hidden_size必须大于0")
                        except ValueError:
                            print("请输入有效数字")
                    
                    while True:
                        try:
                            num_layers = int(input(f"num_hidden_layers (当前{self.teacher_model.config.num_hidden_layers}，建议更小): ") or str(max(6, self.teacher_model.config.num_hidden_layers // 2)))
                            if num_layers > 0:
                                break
                            else:
                                print("num_hidden_layers必须大于0")
                        except ValueError:
                            print("请输入有效数字")
                    
                    while True:
                        try:
                            num_heads = int(input(f"num_attention_heads (当前{self.teacher_model.config.num_attention_heads}，建议更小): ") or str(max(8, self.teacher_model.config.num_attention_heads // 2)))
                            if num_heads > 0 and hidden_size % num_heads == 0:
                                break
                            else:
                                print("num_attention_heads必须大于0且能被hidden_size整除")
                        except ValueError:
                            print("请输入有效数字")
                    
                    # 4. 创建自定义学生模型
                    if not self.create_custom_student_model(hidden_size, num_layers, num_heads):
                        return
                    break
                else:
                    print("无效选择，请重试")
            except ValueError:
                print("请输入有效数字")
            
        # 5. 设置蒸馏参数
        temperature = float(input("请输入蒸馏温度 (默认0.5): ") or "0.5")
        alpha = float(input("请输入蒸馏损失权重 (0-1，默认0.5): ") or "0.5")
        epochs = int(input("请输入训练轮数 (默认3): ") or "3")
        batch_size = int(input("请输入批次大小 (默认4): ") or "4")
        
        # 询问是否限制数据量
        limit_data = input("是否限制数据量进行调参? (y/n，默认n): ").lower()
        max_lines = None
        if limit_data == 'y':
            max_lines = int(input("请输入最大加载行数 (默认500): ") or "500")
        
        # 6. 开始蒸馏
        output_dir = f"distilled_model_{self.selected_model_name}"
        if max_lines:
            output_dir += f"_test{max_lines}"
        
        if not self.distill_model(output_dir, temperature, alpha, epochs, batch_size, max_lines):
            return
            
        print("模型蒸馏管道完成!")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="模型蒸馏脚本")
    parser.add_argument("--model_dir", default="D:\\Model", help="模型文件夹路径")
    parser.add_argument("--data_dir", default="data", help="数据文件夹路径")
    
    args = parser.parse_args()
    
    # 创建蒸馏器
    distiller = ModelDistiller(args.model_dir, args.data_dir)
    
    # 运行蒸馏管道
    distiller.run_distillation_pipeline()

if __name__ == "__main__":
    main() 