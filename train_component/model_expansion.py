import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import json
import glob
from typing import List, Dict, Any
import argparse

class CustomTrainer(Trainer):
    """
    自定义训练器，实现分层学习率
    """
    def __init__(self, original_layers: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_layers = original_layers
        self.step_count = 0
        
    def on_step_begin(self, args, state, control, **kwargs):
        """每步开始时的回调"""
        super().on_step_begin(args, state, control, **kwargs)
        self.step_count += 1
        
        # 每10步打印一次详细内存状态
        if self.step_count % 10 == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                utilization = (allocated / total) * 100
                
                loss_info = f"损失: {state.log_history[-1]['loss']:.4f}" if state.log_history else "损失: N/A"
                print(f"🔄 Step {self.step_count}: {loss_info} | GPU内存: {allocated:.2f}GB/{total:.1f}GB ({utilization:.1f}%) | 保留: {reserved:.2f}GB")
                
                # 内存使用率警告
                if utilization > 85:
                    print(f"⚠️  警告：内存使用率过高 ({utilization:.1f}%)")
                elif utilization > 95:
                    print(f"🚨 危险：内存使用率极高 ({utilization:.1f}%)")
            else:
                loss_info = f"损失: {state.log_history[-1]['loss']:.4f}" if state.log_history else "损失: N/A"
                print(f"🔄 Step {self.step_count}: {loss_info} | CPU模式")
        
    def create_optimizer(self):
        """
        创建分层学习率的优化器
        """
        # 获取所有参数组
        param_groups = []
        
        # 新增层的参数（使用较高学习率）
        new_layer_params = []
        # 原有权重的参数（使用较低学习率）
        original_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 检查是否是新增的层
                if 'layers.' in name:
                    try:
                        layer_num = int(name.split('layers.')[1].split('.')[0])
                        if layer_num >= self.original_layers:
                            new_layer_params.append(param)
                        else:
                            original_params.append(param)
                    except (ValueError, IndexError):
                        # 如果无法解析层号，归类为原有权重
                        original_params.append(param)
                else:
                    # embedding层和输出层使用中等学习率
                    original_params.append(param)
        
        # 设置不同的学习率
        base_lr = self.args.learning_rate
        
        print(f"参数分组统计:")
        print(f"  新增层参数数量: {len(new_layer_params)}")
        print(f"  原有权重参数数量: {len(original_params)}")
        print(f"  总可训练参数数量: {len(new_layer_params) + len(original_params)}")
        
        # 检查参数是否为空
        if len(new_layer_params) == 0 and len(original_params) == 0:
            print("警告：没有找到可训练的参数！")
            print("检查模型参数:")
            total_params = 0
            trainable_params = 0
            for name, param in self.model.named_parameters():
                total_params += 1
                if param.requires_grad:
                    trainable_params += 1
                    print(f"  可训练: {name}")
                else:
                    print(f"  不可训练: {name}")
            print(f"总参数: {total_params}, 可训练参数: {trainable_params}")
        
        if new_layer_params:
            param_groups.append({
                'params': new_layer_params,
                'lr': base_lr * 2,  # 新增层使用2倍学习率
                'name': 'new_layers'
            })
        
        if original_params:
            param_groups.append({
                'params': original_params,
                'lr': base_lr * 0.1,  # 原有权重使用0.1倍学习率
                'name': 'original_layers'
            })
        
        # 创建优化器
        if not param_groups:
            print("警告：没有可训练的参数，使用默认优化器")
            return super().create_optimizer()
        
        try:
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
            print("优化器创建成功")
            return optimizer
        except Exception as e:
            print(f"创建优化器失败: {e}")
            print("使用默认优化器")
            return super().create_optimizer()
        
        print(f"创建分层学习率优化器:")
        print(f"  新增层 ({len(new_layer_params)} 参数): 学习率 {base_lr * 2}")
        print(f"  原有权重 ({len(original_params)} 参数): 学习率 {base_lr * 0.1}")
        
        return optimizer

class ModelExpander:
    def __init__(self, model_dir: str = "../model", data_dir: str = "data"):
        """
        初始化模型扩展器
        
        Args:
            model_dir: 模型文件夹路径（相对于train_component目录）
            data_dir: 数据文件夹路径
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        print(f"模型目录: {os.path.abspath(self.model_dir)}")
        print(f"数据目录: {os.path.abspath(self.data_dir)}")
        
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
        让用户选择要扩展的模型
        
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
                choice = int(input(f"请选择要扩展的模型 (1-{len(models)}): ")) - 1
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
            
            # 使用更节省内存的加载方式
            try:
                # 首先尝试使用device_map自动管理内存
                self.model = AutoModelForCausalLM.from_pretrained(
                    actual_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            except torch.cuda.OutOfMemoryError:
                print("GPU内存不足，尝试使用CPU加载...")
                # 如果GPU内存不足，使用CPU加载
                self.model = AutoModelForCausalLM.from_pretrained(
                    actual_model_path,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                # 然后尝试移动到GPU
                try:
                    self.model = self.model.to(self.device)
                except torch.cuda.OutOfMemoryError:
                    print("GPU内存仍然不足，使用CPU训练")
                    self.device = torch.device("cpu")
                    self.model = self.model.cpu()
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # 记录原始层数
            self.original_layers_count = self.model.config.num_hidden_layers
            
            print(f"模型加载成功，当前参数量: {self.model.num_parameters():,}")
            print(f"使用设备: {self.device}")
            
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
        # 优先使用修复后的数据文件
        fixed_file = os.path.join(self.data_dir, 'fixed_training_data.jsonl')
        if os.path.exists(fixed_file):
            data_files.append(fixed_file)
            print(f"使用修复后的数据文件: {fixed_file}")
        else:
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
                                    text_content = data['text']
                                    # 确保text是字符串
                                    if isinstance(text_content, str):
                                        # 清理文本
                                        cleaned_text = text_content.strip()
                                        # 移除特殊字符
                                        cleaned_text = cleaned_text.replace('\x00', '')
                                        cleaned_text = cleaned_text.replace('\ufffd', '')
                                        if cleaned_text:
                                            texts.append(cleaned_text)
                                    elif isinstance(text_content, list):
                                        # 如果是列表，转换为字符串
                                        text_str = " ".join([str(item) for item in text_content if item])
                                        if text_str.strip():
                                            texts.append(text_str.strip())
                                    else:
                                        text_str = str(text_content)
                                        if text_str.strip():
                                            texts.append(text_str.strip())
                                else:
                                    text_str = str(data)
                                    if text_str.strip():
                                        texts.append(text_str.strip())
                                line_count += 1
                            except json.JSONDecodeError:
                                print(f"跳过无效的JSON行: {line[:100]}...")
                                continue
                            except Exception as e:
                                print(f"处理JSONL行时出错: {e}")
                                continue
                            
        if not texts:
            print("未找到有效训练数据")
            return None
            
        print(f"加载了 {len(texts)} 条训练数据")
        if max_lines:
            print(f"限制加载前 {max_lines} 行数据")
        
        # 确保所有文本都是字符串
        final_texts = []
        for text in texts:
            if isinstance(text, str):
                final_texts.append(text)
            else:
                final_texts.append(str(text))
        
        print(f"最终数据: {len(final_texts)} 条，第一条: {final_texts[0][:100]}...")
        
        # 创建数据集
        dataset = Dataset.from_dict({"text": final_texts})
        return dataset
    

    
    def expand_model(self, target_size: str = None, custom_config: dict = None):
        """
        扩展模型参数量（保留原模型知识）
        
        Args:
            target_size: 目标模型大小 (如 "3b", "7b") 或 None表示自定义
            custom_config: 自定义配置字典
        """
        # 保存原模型状态
        original_model = self.model
        original_config = original_model.config
        self.original_layers_count = original_model.config.num_hidden_layers
        
        if target_size is not None:
            # 使用预设大小
            print(f"开始扩展模型到 {target_size}")
            
            # 根据目标大小调整配置
            size_mapping = {
                "1b": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
                "1.8b": {"hidden_size": 1536, "num_hidden_layers": 30, "num_attention_heads": 12},
                "3b": {"hidden_size": 1536, "num_hidden_layers": 24, "num_attention_heads": 24},
                "7b": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32},
                "9b": {"hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32},
                "soulchat-9b": {"hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32}
            }
            
            if target_size not in size_mapping:
                print(f"不支持的目标大小: {target_size}")
                return False
                
            new_config = size_mapping[target_size]
        else:
            # 使用自定义配置
            print("使用自定义配置扩展模型")
            new_config = custom_config
        
        # 创建新配置
        new_model_config = original_config.__class__.from_pretrained(
            os.path.join(self.model_dir, self.selected_model_name)
        )
        new_model_config.hidden_size = new_config["hidden_size"]
        new_model_config.num_hidden_layers = new_config["num_hidden_layers"]
        new_model_config.num_attention_heads = new_config["num_attention_heads"]
        
        # 正确计算注意力层维度
        head_dim = new_config["hidden_size"] // new_config["num_attention_heads"]
        
        # 保持原有的key_value_heads设置
        if hasattr(original_config, 'num_key_value_heads'):
            new_model_config.num_key_value_heads = original_config.num_key_value_heads
        else:
            new_model_config.num_key_value_heads = new_config["num_attention_heads"]
        
        # 保持其他配置
        new_model_config.hidden_act = getattr(original_config, 'hidden_act', 'silu')
        new_model_config.rope_theta = getattr(original_config, 'rope_theta', 10000.0)
        new_model_config.rms_norm_eps = getattr(original_config, 'rms_norm_eps', 1e-6)
        
        # 保持原有的intermediate_size比例
        if hasattr(original_config, 'intermediate_size'):
            ratio = original_config.intermediate_size / original_config.hidden_size
            new_model_config.intermediate_size = int(new_config["hidden_size"] * ratio)
        else:
            new_model_config.intermediate_size = new_config["hidden_size"] * 4
        
        # 创建新模型（使用CPU初始化以节省GPU内存）
        print("创建扩展后的模型...")
        new_model = AutoModelForCausalLM.from_config(new_model_config)
        # 先在CPU上初始化，避免GPU内存不足
        new_model = new_model.cpu()
        
        # 复制原模型权重到新模型
        print("复制原模型权重...")
        self._copy_weights_preserving_knowledge(original_model, new_model)
        
        # 替换模型并移动到GPU（使用device_map自动管理内存）
        print("将模型移动到GPU...")
        try:
            # 尝试使用device_map自动管理GPU内存
            self.model = new_model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            print("GPU内存不足，尝试使用device_map...")
            try:
                # 使用device_map自动分配内存
                self.model = AutoModelForCausalLM.from_pretrained(
                    None, 
                    config=new_model_config,
                    state_dict=new_model.state_dict(),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"device_map也失败，使用CPU训练: {e}")
                self.model = new_model.cpu()
                self.device = torch.device("cpu")
                print("切换到CPU训练模式")
        
        print(f"模型扩展完成，新参数量: {self.model.num_parameters():,}")
        return True
    
    def _copy_weights_preserving_knowledge(self, original_model, new_model):
        """
        复制权重，保留原模型知识 - 优化版本
        """
        original_state_dict = original_model.state_dict()
        new_state_dict = new_model.state_dict()
        
        # 获取原始和新的配置
        orig_hidden_size = original_model.config.hidden_size
        new_hidden_size = new_model.config.hidden_size
        orig_layers = original_model.config.num_hidden_layers
        new_layers = new_model.config.num_hidden_layers
        
        print(f"🔍 权重复制分析:")
        print(f"  原始hidden_size: {orig_hidden_size}, 新hidden_size: {new_hidden_size}")
        print(f"  原始层数: {orig_layers}, 新层数: {new_layers}")
        print(f"  扩展层数: {new_layers - orig_layers}")
        
        # 1. 复制embedding层
        if 'model.embed_tokens.weight' in original_state_dict and 'model.embed_tokens.weight' in new_state_dict:
            orig_emb = original_state_dict['model.embed_tokens.weight']
            new_emb = new_state_dict['model.embed_tokens.weight']
            
            if orig_hidden_size == new_hidden_size:
                # 维度相同，直接复制
                if orig_emb.shape[0] <= new_emb.shape[0]:
                    new_emb[:orig_emb.shape[0]] = orig_emb
                    print(f"✅ 复制embedding层: {orig_emb.shape} -> {new_emb.shape}")
                else:
                    print(f"⚠️  embedding层词汇表大小不匹配: {orig_emb.shape[0]} > {new_emb.shape[0]}")
            else:
                # 维度不同，使用插值调整
                new_emb = self._resize_embedding(orig_emb, new_hidden_size)
                print(f"🔄 调整embedding层: {orig_emb.shape} -> {new_emb.shape}")
            
            new_state_dict['model.embed_tokens.weight'] = new_emb
        
        # 2. 复制transformer层 - 保持原有知识
        copy_layers = min(orig_layers, new_layers)
        copied_params = 0
        skipped_params = 0
        
        print(f"📋 开始复制transformer层...")
        for i in range(copy_layers):
            layer_copied = 0
            for key in original_state_dict.keys():
                if f'.layers.{i}.' in key:
                    new_key = key.replace(f'.layers.{i}.', f'.layers.{i}.')
                    if new_key in new_state_dict:
                        orig_param = original_state_dict[key]
                        new_param = new_state_dict[new_key]
                        
                        # 检查维度是否匹配
                        if orig_param.shape == new_param.shape:
                            new_state_dict[new_key] = orig_param
                            copied_params += 1
                            layer_copied += 1
                        else:
                            # 维度不匹配，尝试智能调整
                            if self._can_resize_parameter(orig_param, new_param):
                                resized_param = self._resize_parameter(orig_param, new_param.shape)
                                new_state_dict[new_key] = resized_param
                                copied_params += 1
                                layer_copied += 1
                                print(f"🔄 调整参数维度: {key} {orig_param.shape} -> {new_param.shape}")
                            else:
                                print(f"⚠️  跳过维度不匹配的参数: {key} {orig_param.shape} -> {new_param.shape}")
                                skipped_params += 1
            
            if layer_copied > 0:
                print(f"  ✅ 层 {i}: 复制了 {layer_copied} 个参数")
        
        print(f"📊 权重复制统计:")
        print(f"  复制层数: {copy_layers}/{orig_layers}")
        print(f"  成功复制参数: {copied_params}")
        print(f"  跳过参数: {skipped_params}")
        
        # 3. 复制输出层
        norm_copied = 0
        for norm_key in ['model.norm.weight', 'model.norm.bias']:
            if norm_key in original_state_dict and norm_key in new_state_dict:
                new_state_dict[norm_key] = original_state_dict[norm_key]
                norm_copied += 1
        if norm_copied > 0:
            print(f"✅ 复制输出归一化层: {norm_copied} 个参数")
        
        # 4. 复制lm_head
        if 'lm_head.weight' in original_state_dict and 'lm_head.weight' in new_state_dict:
            orig_lm_head = original_state_dict['lm_head.weight']
            new_lm_head = new_state_dict['lm_head.weight']
            
            if orig_lm_head.shape[0] <= new_lm_head.shape[0]:
                new_lm_head[:orig_lm_head.shape[0]] = orig_lm_head
                print(f"✅ 复制lm_head: {orig_lm_head.shape} -> {new_lm_head.shape}")
            else:
                print(f"⚠️  lm_head词汇表大小不匹配: {orig_lm_head.shape[0]} > {new_lm_head.shape[0]}")
        
        # 5. 加载权重到新模型
        print("🔄 加载权重到新模型...")
        missing_keys, unexpected_keys = new_model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  缺失的键: {len(missing_keys)} 个")
        if unexpected_keys:
            print(f"⚠️  意外的键: {len(unexpected_keys)} 个")
        
        # 6. 智能初始化新增的层
        if new_layers > orig_layers:
            print(f"🧠 智能初始化新增的层 {orig_layers} 到 {new_layers-1}...")
            self._initialize_new_layers_smart(new_model, orig_layers, new_layers)
        
        print("✅ 权重复制完成！")
    
    def _can_resize_parameter(self, orig_param, new_param):
        """检查参数是否可以调整大小"""
        # 只允许调整某些类型的参数
        resizable_types = ['weight']
        param_name = getattr(new_param, 'name', '')
        return any(t in param_name for t in resizable_types)
    
    def _resize_parameter(self, orig_param, new_shape):
        """调整参数大小"""
        if len(orig_param.shape) == 2 and len(new_shape) == 2:
            # 线性层权重
            return torch.nn.functional.interpolate(
                orig_param.unsqueeze(0), 
                size=new_shape, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        else:
            # 其他参数，使用零填充或截断
            new_param = torch.zeros(new_shape, device=orig_param.device, dtype=orig_param.dtype)
            if len(orig_param.shape) == len(new_shape):
                slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(orig_param.shape, new_shape))
                new_param[slices] = orig_param[slices]
            return new_param
    
    def _resize_embedding(self, embedding, new_hidden_size):
        """
        调整embedding层的维度
        """
        vocab_size, hidden_size = embedding.shape
        
        if hidden_size == new_hidden_size:
            return embedding
        
        # 创建新的embedding
        new_embedding = torch.zeros(vocab_size, new_hidden_size, device=embedding.device, dtype=embedding.dtype)
        
        if hidden_size > new_hidden_size:
            # 从大到小：使用平均池化
            scale_factor = hidden_size // new_hidden_size
            for i in range(new_hidden_size):
                start_idx = i * scale_factor
                end_idx = min((i + 1) * scale_factor, hidden_size)
                new_embedding[:, i] = embedding[:, start_idx:end_idx].mean(dim=1)
        else:
            # 从小到大：使用插值
            new_embedding = torch.nn.functional.interpolate(
                embedding.unsqueeze(0).transpose(1, 2),  # [1, hidden_size, vocab_size]
                size=new_hidden_size,
                mode='linear',
                align_corners=False
            ).squeeze(0).transpose(0, 1)  # [vocab_size, new_hidden_size]
        
        return new_embedding
    
    def _initialize_new_layers_smart(self, model, start_layer, end_layer):
        """
        智能初始化新增的层 - 使用渐进式初始化策略
        """
        print(f"🧠 智能初始化新增的层 {start_layer} 到 {end_layer-1}")
        
        if start_layer >= end_layer:
            print("⚠️  没有新增层需要初始化")
            return
        
        # 获取参考层（使用最后一层作为参考）
        reference_layer = model.model.layers[start_layer - 1]
        print(f"📋 使用层 {start_layer-1} 作为初始化参考")
        
        # 计算初始化策略
        total_new_layers = end_layer - start_layer
        print(f"📊 需要初始化 {total_new_layers} 个新层")
        
        for i in range(start_layer, end_layer):
            current_layer = model.model.layers[i]
            layer_index = i - start_layer
            
            # 计算初始化权重（越后面的层，权重越接近参考层）
            if total_new_layers > 1:
                weight_factor = layer_index / (total_new_layers - 1)
            else:
                weight_factor = 1.0
            
            print(f"  🔧 初始化层 {i} (权重因子: {weight_factor:.2f})")
            
            # 1. 初始化注意力层
            if hasattr(current_layer.self_attn, 'q_proj') and hasattr(reference_layer.self_attn, 'q_proj'):
                # 使用参考层权重 + 小随机噪声
                noise_scale = 0.01 * (1 - weight_factor)  # 越后面的层噪声越小
                
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    ref_proj = getattr(reference_layer.self_attn, proj_name)
                    cur_proj = getattr(current_layer.self_attn, proj_name)
                    
                    if hasattr(ref_proj, 'weight') and hasattr(cur_proj, 'weight'):
                        # 复制权重并添加噪声
                        cur_proj.weight.data = ref_proj.weight.data.clone()
                        if noise_scale > 0:
                            noise = torch.randn_like(cur_proj.weight.data) * noise_scale
                            cur_proj.weight.data += noise
                        
                        # 复制偏置（如果存在）
                        if hasattr(ref_proj, 'bias') and hasattr(cur_proj, 'bias') and ref_proj.bias is not None:
                            cur_proj.bias.data = ref_proj.bias.data.clone()
                            if noise_scale > 0:
                                noise = torch.randn_like(cur_proj.bias.data) * noise_scale
                                cur_proj.bias.data += noise
            
            # 2. 初始化MLP层
            if hasattr(current_layer.mlp, 'gate_proj') and hasattr(reference_layer.mlp, 'gate_proj'):
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    ref_proj = getattr(reference_layer.mlp, proj_name)
                    cur_proj = getattr(current_layer.mlp, proj_name)
                    
                    if hasattr(ref_proj, 'weight') and hasattr(cur_proj, 'weight'):
                        # 复制权重并添加噪声
                        cur_proj.weight.data = ref_proj.weight.data.clone()
                        if noise_scale > 0:
                            noise = torch.randn_like(cur_proj.weight.data) * noise_scale
                            cur_proj.weight.data += noise
                        
                        # 复制偏置（如果存在）
                        if hasattr(ref_proj, 'bias') and hasattr(cur_proj, 'bias') and ref_proj.bias is not None:
                            cur_proj.bias.data = ref_proj.bias.data.clone()
                            if noise_scale > 0:
                                noise = torch.randn_like(cur_proj.bias.data) * noise_scale
                                cur_proj.bias.data += noise
            
            # 3. 初始化层归一化
            for norm_name in ['input_layernorm', 'post_attention_layernorm']:
                if hasattr(current_layer, norm_name) and hasattr(reference_layer, norm_name):
                    ref_norm = getattr(reference_layer, norm_name)
                    cur_norm = getattr(current_layer, norm_name)
                    
                    # 复制权重
                    if hasattr(ref_norm, 'weight') and hasattr(cur_norm, 'weight'):
                        cur_norm.weight.data = ref_norm.weight.data.clone()
                    
                    # 复制偏置（RMSNorm没有bias，LayerNorm有bias）
                    if hasattr(ref_norm, 'bias') and hasattr(cur_norm, 'bias') and ref_norm.bias is not None:
                        cur_norm.bias.data = ref_norm.bias.data.clone()
        
        print(f"✅ 智能初始化完成！")
        print(f"📈 初始化策略:")
        print(f"  - 使用参考层权重作为基础")
        print(f"  - 添加渐进式随机噪声")
        print(f"  - 保持原有知识的同时增加多样性")
    
    def _initialize_new_layers(self, model, start_layer, end_layer):
        """
        初始化新增的层（保留原方法作为备用）
        """
        print(f"初始化新增的层 {start_layer} 到 {end_layer-1}")
        
        # 使用原模型的最后一层作为初始化参考
        if start_layer > 0:
            reference_layer = model.model.layers[start_layer - 1]
            
            for i in range(start_layer, end_layer):
                current_layer = model.model.layers[i]
                
                # 复制注意力层权重
                if hasattr(current_layer.self_attn, 'q_proj') and hasattr(reference_layer.self_attn, 'q_proj'):
                    current_layer.self_attn.q_proj.weight.data = reference_layer.self_attn.q_proj.weight.data.clone()
                    current_layer.self_attn.k_proj.weight.data = reference_layer.self_attn.k_proj.weight.data.clone()
                    current_layer.self_attn.v_proj.weight.data = reference_layer.self_attn.v_proj.weight.data.clone()
                    current_layer.self_attn.o_proj.weight.data = reference_layer.self_attn.o_proj.weight.data.clone()
                
                # 复制MLP层权重
                if hasattr(current_layer.mlp, 'gate_proj') and hasattr(reference_layer.mlp, 'gate_proj'):
                    current_layer.mlp.gate_proj.weight.data = reference_layer.mlp.gate_proj.weight.data.clone()
                    current_layer.mlp.up_proj.weight.data = reference_layer.mlp.up_proj.weight.data.clone()
                    current_layer.mlp.down_proj.weight.data = reference_layer.mlp.down_proj.weight.data.clone()
                
                # 复制层归一化权重（兼容RMSNorm和LayerNorm）
                if hasattr(current_layer, 'input_layernorm') and hasattr(reference_layer, 'input_layernorm'):
                    current_layer.input_layernorm.weight.data = reference_layer.input_layernorm.weight.data.clone()
                    # RMSNorm没有bias，LayerNorm有bias
                    if hasattr(current_layer.input_layernorm, 'bias') and hasattr(reference_layer.input_layernorm, 'bias'):
                        current_layer.input_layernorm.bias.data = reference_layer.input_layernorm.bias.data.clone()
                
                if hasattr(current_layer, 'post_attention_layernorm') and hasattr(reference_layer, 'post_attention_layernorm'):
                    current_layer.post_attention_layernorm.weight.data = reference_layer.post_attention_layernorm.weight.data.clone()
                    # RMSNorm没有bias，LayerNorm有bias
                    if hasattr(current_layer.post_attention_layernorm, 'bias') and hasattr(reference_layer.post_attention_layernorm, 'bias'):
                        current_layer.post_attention_layernorm.bias.data = reference_layer.post_attention_layernorm.bias.data.clone()
    
    def _clear_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            # 更激进的内存清理
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 清理PyTorch缓存
            if hasattr(torch.cuda, 'memory_summary'):
                print("清理前内存状态:")
                print(torch.cuda.memory_summary(device=0, abbreviated=True))
            
            print("GPU内存已清理")
            
            # 显示清理后的内存状态
            if torch.cuda.is_available():
                print(f"清理后内存状态:")
                print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                print(f"  已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
                print(f"  总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def _optimize_memory_for_small_gpu(self):
        """火力全开GPU优化设置"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 检测到怪兽级GPU: {total_memory:.1f} GB")
            
            # 设置环境变量 - 火力全开配置
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
            
            # 火力全开内存使用
            memory_fraction = 0.95  # 使用95%内存
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            print(f"🔥 火力全开模式: 使用 {memory_fraction:.1%} GPU内存")
            print(f"💪 可用内存: {total_memory * memory_fraction:.1f} GB")
            
            # 根据GPU大小设置不同的优化策略
            if total_memory >= 100:  # 100GB+
                print("🎯 怪兽级GPU配置: 最大批次 + 最快训练")
            elif total_memory >= 50:  # 50GB+
                print("⚡ 高性能GPU配置: 大批次 + 高效训练")
            else:
                print("🚀 标准高性能配置")
    
    def _monitor_gpu_memory(self):
        """监控GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            
            print(f"🔍 GPU内存监控: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB, 可用 {free:.2f}GB / 总计 {total:.1f}GB")
            
            if free < 0.5:  # 如果可用内存少于500MB
                print("⚠️  警告：GPU内存不足，建议清理内存")
                self._clear_gpu_memory()
    
    def _print_memory_status(self, stage=""):
        """打印当前内存状态"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            utilization = (allocated / total) * 100
            
            stage_info = f"[{stage}] " if stage else ""
            print(f"💾 {stage_info}GPU内存: {allocated:.2f}GB/{total:.1f}GB ({utilization:.1f}%) | 可用: {free:.2f}GB")
        else:
            print(f"💾 {stage}CPU模式")
    
    def _create_gradient_accumulation_trainer(self, train_dataset, output_dir, epochs, batch_size, gradient_accumulation_steps):
        """
        创建梯度累积训练器
        
        技术要点：
        1. 小批次训练：使用较小的batch_size减少内存使用
        2. 梯度累积：通过gradient_accumulation_steps累积梯度，模拟大批次训练
        3. 等效大批次 = batch_size × gradient_accumulation_steps
        4. 内存使用 = 小批次内存 × 1，而不是大批次内存
        5. 训练效果：与大批次训练等效，但内存使用更少
        
        示例：
        - batch_size=1, gradient_accumulation_steps=32 → 等效大批次32
        - batch_size=2, gradient_accumulation_steps=16 → 等效大批次32
        - 内存使用：只需要1或2个样本的内存，而不是32个样本的内存
        """
        print(f"创建梯度累积训练器...")
        print(f"批次大小: {batch_size}, 梯度累积步数: {gradient_accumulation_steps}")
        print(f"等效大批次: {batch_size * gradient_accumulation_steps}")
        print(f"内存优势: 只需要 {batch_size} 个样本的内存，而不是 {batch_size * gradient_accumulation_steps} 个样本的内存")
        
        # 根据GPU大小智能设置训练参数
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if total_memory >= 100:  # 140GB怪兽级GPU
                print("🎯 怪兽级GPU训练参数 - 火力全开!")
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    save_steps=500,  # 更频繁保存
                    save_total_limit=5,  # 保存更多checkpoint
                    logging_steps=10,  # 更频繁日志
                    learning_rate=3e-5,  # 稍高学习率
                    warmup_steps=1000,
                    weight_decay=0.01,
                    logging_dir=f"{output_dir}/logs",
                    remove_unused_columns=False,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    fp16=True,  # 混合精度
                    dataloader_pin_memory=True,
                    gradient_checkpointing=False,  # 关闭以提升速度
                    optim="adamw_torch_fused",  # 融合优化器
                    max_grad_norm=1.0,
                    dataloader_num_workers=8,  # 更多数据加载进程
                    group_by_length=True,
                    dataloader_drop_last=False,
                    dataloader_prefetch_factor=4,  # 更多预取
                    torch_compile=True,  # 启用编译优化
                )
            elif total_memory >= 50:  # 50GB+高性能GPU
                print("⚡ 高性能GPU训练参数")
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    save_steps=1000,
                    save_total_limit=3,
                    logging_steps=50,
                    learning_rate=2e-5,
                    warmup_steps=500,
                    weight_decay=0.01,
                    logging_dir=f"{output_dir}/logs",
                    remove_unused_columns=False,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    fp16=True,
                    dataloader_pin_memory=True,
                    gradient_checkpointing=False,
                    optim="adamw_torch_fused",
                    max_grad_norm=1.0,
                    dataloader_num_workers=4,
                    group_by_length=True,
                    dataloader_drop_last=False,
                    dataloader_prefetch_factor=2,
                    torch_compile=True,
                )
            else:
                # 标准高性能配置
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    save_steps=1000,
                    save_total_limit=3,
                    logging_steps=50,
                    learning_rate=2e-5,
                    warmup_steps=500,
                    weight_decay=0.01,
                    logging_dir=f"{output_dir}/logs",
                    remove_unused_columns=False,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    fp16=True,
                    dataloader_pin_memory=True,
                    gradient_checkpointing=False,
                    optim="adamw_torch_fused",
                    max_grad_norm=1.0,
                    dataloader_num_workers=2,
                    group_by_length=True,
                    dataloader_drop_last=False,
                    dataloader_prefetch_factor=2,
                    torch_compile=False,
                )
        else:
            # CPU训练设置
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=1000,
                save_total_limit=3,
                logging_steps=50,
                learning_rate=2e-5,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                remove_unused_columns=False,
                gradient_accumulation_steps=gradient_accumulation_steps,
                fp16=False,
                dataloader_pin_memory=False,
                gradient_checkpointing=False,
                optim="adamw_torch",
                max_grad_norm=1.0,
                dataloader_num_workers=0,
                group_by_length=True,
                dataloader_drop_last=False,
                dataloader_prefetch_factor=None,
                torch_compile=False,
            )
        
        from transformers import DataCollatorForLanguageModeling
        
        # 创建数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        return trainer
    

    
    def tokenize_dataset(self, max_lines: int = None):
        """
        分词处理数据集（与训练分离）
        
        Args:
            max_lines: 最大加载行数，None表示加载全部
            
        Returns:
            分词后的数据集
        """
        import time
        start_time = time.time()
        
        # 清理GPU内存
        self._clear_gpu_memory()
        
        # 加载训练数据
        dataset = self.load_training_data(max_lines)
        if dataset is None:
            return None
            
        # 分词
        print("开始分词处理...")
        print(f"数据集大小: {len(dataset)}")
        
        try:
            # 直接处理数据，不使用map方法
            texts = dataset['text']
            print(f"开始处理 {len(texts)} 条文本...")
            
            valid_data = []
            for i, text in enumerate(texts):
                try:
                    # 清理文本
                    if isinstance(text, str):
                        cleaned_text = text.strip()
                        cleaned_text = cleaned_text.replace('\x00', '')
                        cleaned_text = cleaned_text.replace('\ufffd', '')
                        if len(cleaned_text) > 2000:
                            cleaned_text = cleaned_text[:2000]
                        
                        if cleaned_text:
                            # 根据GPU内存设置序列长度
                            if torch.cuda.is_available():
                                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                                if total_memory >= 100:  # 140GB怪兽级GPU
                                    max_length = 2048  # 超长序列
                                elif total_memory >= 50:  # 50GB+高性能GPU
                                    max_length = 1024  # 长序列
                                else:
                                    max_length = 512  # 标准长度
                            else:
                                max_length = 256  # CPU使用中等长度
                            
                            # 直接分词
                            tokenized = self.tokenizer(
                                cleaned_text,
                                truncation=True,
                                padding=False,
                                max_length=max_length,
                                return_tensors=None
                            )
                            
                            input_ids = tokenized['input_ids']
                            attention_mask = tokenized['attention_mask']
                            
                            # 确保是列表格式
                            if isinstance(input_ids, torch.Tensor):
                                input_ids = input_ids.tolist()
                            if isinstance(attention_mask, torch.Tensor):
                                attention_mask = attention_mask.tolist()
                            
                            # 确保是整数列表
                            input_ids = [int(x) for x in input_ids if x is not None]
                            attention_mask = [int(x) for x in attention_mask if x is not None]
                            
                            if len(input_ids) == len(attention_mask) and len(input_ids) > 0:
                                valid_data.append({
                                    'input_ids': input_ids,
                                    'attention_mask': attention_mask
                                })
                                
                                # 每处理10条显示一次进度
                                if (i + 1) % 10 == 0:
                                    print(f"已处理 {i + 1}/{len(texts)} 条文本")
                    
                except Exception as e:
                    print(f"处理第{i}条文本时出错: {e}")
                    continue
            
            end_time = time.time()
            print(f"分词处理完成，耗时: {end_time - start_time:.2f} 秒")
            print(f"成功处理 {len(valid_data)} 条有效数据")
            
            if not valid_data:
                print("错误：没有有效的数据样本")
                return None
            
            # 检查分词结果
            if len(valid_data) > 0:
                sample = valid_data[0]
                print(f"分词样本检查:")
                print(f"  - input_ids长度: {len(sample['input_ids'])}")
                print(f"  - attention_mask长度: {len(sample['attention_mask'])}")
                print(f"  - 前10个token: {sample['input_ids'][:10]}")
                
                # 解码前几个token看看内容
                try:
                    decoded_text = self.tokenizer.decode(sample['input_ids'][:20])
                    print(f"  - 解码前20个token: {decoded_text}")
                except Exception as e:
                    print(f"  - 解码失败: {e}")
            
            # 创建新的数据集
            from datasets import Dataset
            tokenized_dataset = Dataset.from_list(valid_data)
            return tokenized_dataset
            
        except Exception as e:
            print(f"分词处理出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_expanded_model(self, output_dir: str, epochs: int = 3, batch_size: int = 4, tokenized_dataset=None):
        """
        训练扩展后的模型
        
        Args:
            output_dir: 输出目录
            epochs: 训练轮数
            batch_size: 批次大小
            tokenized_dataset: 已经分词的数据集，如果为None则重新分词
        """
        import time
        start_time = time.time()
        
        # 如果没有提供分词后的数据集，则进行分词
        if tokenized_dataset is None:
            print("未提供分词后的数据集，需要重新分词...")
            return False
        
        # 应用标准内存优化
        self._optimize_memory_for_small_gpu()
        
        # 使用全部数据作为训练集
        train_dataset = tokenized_dataset
        print(f"训练集大小: {len(train_dataset)}")
        

        
        # 检查GPU内存
        if torch.cuda.is_available():
            print(f"GPU内存使用情况:")
            print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"  已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            print(f"  总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 根据GPU大小智能设置训练参数
        gradient_accumulation_steps = 8  # 默认值
        max_length = 512  # 默认值
        
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if total_memory >= 100:  # 140GB怪兽级GPU
                print("🎯 怪兽级GPU配置 - 火力全开!")
                batch_size = 32  # 大批次
                gradient_accumulation_steps = 4  # 少累积，多并行
                max_length = 2048  # 长序列
                print(f"🔥 批次大小: {batch_size}")
                print(f"🔥 梯度累积步数: {gradient_accumulation_steps}")
                print(f"🔥 等效大批次: {batch_size * gradient_accumulation_steps}")
                print(f"🔥 序列长度: {max_length}")
            elif total_memory >= 50:  # 50GB+高性能GPU
                print("⚡ 高性能GPU配置")
                batch_size = 16
                gradient_accumulation_steps = 8
                max_length = 1024
                print(f"⚡ 批次大小: {batch_size}")
                print(f"⚡ 梯度累积步数: {gradient_accumulation_steps}")
                print(f"⚡ 等效大批次: {batch_size * gradient_accumulation_steps}")
                print(f"⚡ 序列长度: {max_length}")
            else:
                print("🚀 标准高性能配置")
                batch_size = 8
                gradient_accumulation_steps = 8
                max_length = 512
                print(f"🚀 批次大小: {batch_size}")
                print(f"🚀 梯度累积步数: {gradient_accumulation_steps}")
                print(f"🚀 等效大批次: {batch_size * gradient_accumulation_steps}")
                print(f"🚀 序列长度: {max_length}")
        else:
            print("使用标准训练设置")
            batch_size = 2
            gradient_accumulation_steps = 8
            max_length = 512
        
        # 检查模型大小
        model_params = self.model.num_parameters()
        print(f"模型参数量: {model_params:,}")
        print(f"原始层数: {self.original_layers_count}")
        
        print(f"原始层数: {self.original_layers_count}")
        

        
        # 使用梯度累积训练器
        print("创建梯度累积训练器...")
        
        trainer = self._create_gradient_accumulation_trainer(
            train_dataset, output_dir, epochs, batch_size, gradient_accumulation_steps
        )
        print("梯度累积训练器创建完成")
        

        
        # 开始训练
        print("开始训练扩展后的模型...")
        print(f"训练参数:")
        print(f"  - 批次大小: {batch_size}")
        print(f"  - 训练轮数: {epochs}")
        print(f"  - 学习率: {trainer.args.learning_rate}")
        print(f"  - 梯度累积步数: {trainer.args.gradient_accumulation_steps}")
        print(f"  - 等效大批次: {batch_size * gradient_accumulation_steps}")
        print(f"  - 使用设备: {self.device}")
        print(f"  - 训练数据量: {len(train_dataset)}")
        
        print("="*50)
        print("开始训练...")
        print("="*50)
        
        try:
            trainer.train()
            print("训练完成！")
            

        except torch.cuda.OutOfMemoryError as e:
            print(f"GPU内存不足: {e}")
            print("训练失败，请检查GPU内存或减少模型大小")
            return False
        except Exception as e:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 保存模型
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"扩展训练完成，模型已保存到: {output_dir}")
        return True
    
    def run_expansion_pipeline(self):
        """
        运行完整的模型扩展流程
        """
        print("=== 模型扩展训练管道 ===")
        
        # 1. 选择模型
        self.selected_model_name = self.select_model()
        if self.selected_model_name is None:
            return
            
        # 2. 加载模型
        if not self.load_model_and_tokenizer(self.selected_model_name):
            return
            
        # 3. 智能推荐扩展配置
        print(f"\n🔍 智能分析模型: {self.selected_model_name}")
        
        # 分析当前模型配置
        current_config = self.model.config
        current_params = self.model.num_parameters() / 1e9
        print(f"📊 当前模型信息:")
        print(f"  - 模型类型: {getattr(current_config, 'model_type', 'unknown')}")
        print(f"  - 隐藏层大小: {current_config.hidden_size}")
        print(f"  - 层数: {current_config.num_hidden_layers}")
        print(f"  - 注意力头数: {current_config.num_attention_heads}")
        print(f"  - 参数量: {current_params:.2f}B")
        
        # 智能推荐
        recommended_size = None
        if "soulchat" in self.selected_model_name.lower() and current_params >= 7.5:
            recommended_size = "soulchat-9b"
            print(f"🎯 推荐配置: {recommended_size} (SoulChat专用优化)")
        elif current_params >= 7.0:
            recommended_size = "9b"
            print(f"🎯 推荐配置: {recommended_size} (大模型扩展)")
        elif current_params >= 3.0:
            recommended_size = "7b"
            print(f"🎯 推荐配置: {recommended_size} (中等模型扩展)")
        else:
            recommended_size = "3b"
            print(f"🎯 推荐配置: {recommended_size} (小模型扩展)")
        
        print("\n请选择扩展方式:")
        print("1. 使用预设大小")
        print("2. 自定义参数")
        if recommended_size:
            print(f"3. 使用推荐配置 ({recommended_size})")
        
        while True:
            try:
                choice = int(input("请选择 (1-2): "))
                if choice == 1:
                    # 使用预设大小
                    print("\n可用的扩展大小:")
                    sizes = ["1b", "1.8b", "3b", "7b", "9b", "soulchat-9b"]
                    for i, size in enumerate(sizes, 1):
                        print(f"{i}. {size}")
                        
                    while True:
                        try:
                            size_choice = int(input(f"请选择目标大小 (1-{len(sizes)}): ")) - 1
                            if 0 <= size_choice < len(sizes):
                                target_size = sizes[size_choice]
                                custom_config = None
                                break
                            else:
                                print("无效选择，请重试")
                        except ValueError:
                            print("请输入有效数字")
                    break
                elif choice == 2:
                    # 自定义参数
                    print("\n请输入自定义参数:")
                    
                    while True:
                        try:
                            hidden_size = int(input("hidden_size (默认512): ") or "512")
                            if hidden_size > 0:
                                break
                            else:
                                print("hidden_size必须大于0")
                        except ValueError:
                            print("请输入有效数字")
                    
                    while True:
                        try:
                            num_layers = int(input("num_hidden_layers (默认6): ") or "6")
                            if num_layers > 0:
                                break
                            else:
                                print("num_hidden_layers必须大于0")
                        except ValueError:
                            print("请输入有效数字")
                    
                    while True:
                        try:
                            num_heads = int(input("num_attention_heads (默认8): ") or "8")
                            if num_heads > 0 and hidden_size % num_heads == 0:
                                break
                            else:
                                print("num_attention_heads必须大于0且能被hidden_size整除")
                        except ValueError:
                            print("请输入有效数字")
                    
                    target_size = None
                    custom_config = {
                        "hidden_size": hidden_size,
                        "num_hidden_layers": num_layers,
                        "num_attention_heads": num_heads
                    }
                    break
                elif choice == 3 and recommended_size:
                    # 使用推荐配置
                    target_size = recommended_size
                    custom_config = None
                    print(f"✅ 使用推荐配置: {recommended_size}")
                    break
                else:
                    print("无效选择，请重试")
            except ValueError:
                print("请输入有效数字")
                
        # 4. 扩展模型
        if not self.expand_model(target_size, custom_config):
            return
                
        # 5. 设置训练参数
        epochs = int(input("请输入训练轮数 (默认2): ") or "2")
        batch_size = int(input("请输入批次大小 (默认2): ") or "2")
        
        # 询问是否限制数据量
        limit_data = input("是否限制数据量进行调参? (y/n，默认n): ").lower()
        max_lines = None
        if limit_data == 'y':
            max_lines = int(input("请输入最大加载行数 (默认500): ") or "500")
        
        # 6. 先进行分词处理
        print("\n=== 分词处理阶段 ===")
        tokenized_dataset = self.tokenize_dataset(max_lines)
        if tokenized_dataset is None:
            print("分词失败，无法继续")
            return
        
        # 询问用户是否开始训练
        print("\n=== 训练确认阶段 ===")
        print(f"分词完成，训练集大小: {len(tokenized_dataset)}")
        
        # 检查GPU内存
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU总内存: {total_memory:.1f} GB")
            if total_memory < 8:
                print("警告：GPU内存较小，建议使用更小的模型或更小的批次大小")
                print("建议：")
                print("  1. 使用更小的模型（如1b而不是1.8b）")
                print("  2. 使用更小的批次大小（batch_size=1）")
                print("  3. 减少训练数据量")
        
        start_training = input("是否开始训练？(y/n): ").lower().strip()
        if start_training != 'y':
            print("用户取消训练")
            return
        
        # 7. 开始训练
        # 创建trained目录
        trained_dir = os.path.join(self.model_dir, "trained")
        os.makedirs(trained_dir, exist_ok=True)
        
        # 使用固定的模型名称
        output_dir = os.path.join(trained_dir, "chuxin1.0")
        
        print(f"训练后的模型将保存到: {output_dir}")
        
        if not self.train_expanded_model(output_dir, epochs, batch_size, tokenized_dataset):
            return
            
        print("模型扩展训练管道完成!")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="模型扩展训练脚本")
    parser.add_argument("--model_dir", default="../model", help="模型文件夹路径（相对于train_component目录）")
    parser.add_argument("--data_dir", default="data", help="数据文件夹路径")
    
    args = parser.parse_args()
    
    # 创建扩展器
    expander = ModelExpander(args.model_dir, args.data_dir)
    
    # 运行扩展管道
    expander.run_expansion_pipeline()

if __name__ == "__main__":
    main() 