import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, Trainer
from datasets import Dataset
import json
import glob
from typing import List, Dict, Any
import argparse
from pathlib import Path

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
    def __init__(self, model_dir: str = "model", data_dir: str = "data"):
        """
        初始化模型扩展器
        
        Args:
            model_dir: 模型文件夹路径（相对于train_component目录）
            data_dir: 数据文件夹路径
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        # 强制检查GPU可用性
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，无法使用GPU训练！")
            print("请检查：")
            print("1. 是否安装了CUDA版本的PyTorch")
            print("2. 是否有可用的GPU")
            print("3. CUDA驱动是否正确安装")
            raise RuntimeError("CUDA不可用，无法进行GPU训练")
        
        self.device = torch.device("cuda")
        print(f"✅ 使用设备: {self.device}")
        print(f"🎮 GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"模型目录: {os.path.abspath(self.model_dir)}")
        print(f"数据目录: {os.path.abspath(self.data_dir)}")
        
    def list_models(self) -> List[str]:
        """
        列出模型文件夹中的所有模型 - 完全照抄model_downloader.py的方式
        
        Returns:
            模型路径列表
        """
        print("📚 可用的模型:")
        print("=" * 40)
        
        if not os.path.exists(self.model_dir):
            print("❌ 模型目录不存在")
            return []
            
        print(f"🔍 扫描目录: {self.model_dir}")
        print(f"🔍 绝对路径: {os.path.abspath(self.model_dir)}")
        
        models = []
        model_dir_path = Path(self.model_dir)
        
        # 列出所有目录项
        all_items = list(model_dir_path.iterdir())
        print(f"📋 发现 {len(all_items)} 个目录项:")
        for item in all_items:
            print(f"   - {item.name} ({'目录' if item.is_dir() else '文件'})")
        
        for i, model_path in enumerate(all_items, 1):
            if model_path.is_dir():
                print(f"\n🔍 检查目录 {i}: {model_path.name}")
                
                # 过滤掉训练输出目录
                if model_path.name in ['trained', 'output', 'checkpoints', 'logs']:
                    print(f"   ⏭️  跳过训练目录: {model_path.name}")
                    continue
                    
                # 列出目录内容
                try:
                    dir_contents = list(model_path.iterdir())
                    print(f"   📁 目录内容: {[f.name for f in dir_contents[:10]]}...")
                except Exception as e:
                    print(f"   ❌ 无法读取目录内容: {e}")
                    continue
                
                info_file = model_path / "model_info.json"
                if info_file.exists():
                    print(f"   ✅ 找到model_info.json")
                    try:
                        with open(info_file, "r", encoding="utf-8") as f:
                            info = json.load(f)
                        print(f"{i}. 📁 {model_path.name}")
                        print(f"   原始名称: {info.get('name', 'Unknown')}")
                        print(f"   下载源: {info.get('source', 'Unknown')}")
                        print(f"   下载时间: {info.get('download_time', 'Unknown')}")
                        
                        # 计算模型大小
                        size = self.get_model_size(model_path)
                        print(f"   大小: {size}")
                        
                        # 显示详细模型信息
                        self.show_model_details(model_path)
                        
                        models.append(str(model_path))
                    except Exception as e:
                        print(f"   ❌ 读取model_info.json失败: {e}")
                        print(f"{i}. 📁 {model_path.name} (信息文件损坏)")
                        models.append(str(model_path))
                else:
                    print(f"   ⚠️  未找到model_info.json")
                    # 检查是否有config.json文件来确认是真正的模型
                    config_file = model_path / "config.json"
                    if config_file.exists():
                        print(f"   ✅ 找到config.json，认为是模型")
                        
                        # 检查是否有权重文件
                        weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
                        if weight_files:
                            print(f"   ✅ 找到 {len(weight_files)} 个权重文件")
                            print(f"{i}. 📁 {model_path.name} (无信息文件)")
                            # 尝试显示模型详细信息
                            self.show_model_details(model_path)
                            models.append(str(model_path))
                        else:
                            print(f"   ❌ 未找到权重文件")
                            print(f"{i}. ⏭️  跳过不完整模型: {model_path.name}")
                            continue
                    else:
                        print(f"   ❌ 未找到config.json")
                        print(f"{i}. ⏭️  跳过非模型目录: {model_path.name}")
                    continue
        
        if not models:
            print("\n❌ 未找到任何模型")
            print("💡 提示:")
            print("1. 确保模型目录包含有效的模型文件")
            print("2. 模型目录应该包含 config.json 文件")
            print("3. 可以使用 model_chat.py 来测试模型是否可用")
        else:
            print(f"\n✅ 找到 {len(models)} 个模型")
                    
        return models
    
    def show_model_details(self, model_path: Path):
        """显示模型的详细信息 - 完全照抄model_downloader.py"""
        try:
            print(f"   🔍 正在分析模型信息...")
            
            # 尝试加载配置
            config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
            
            print(f"   📊 模型配置:")
            print(f"     模型类型: {getattr(config, 'model_type', 'unknown')}")
            print(f"     隐藏层大小: {getattr(config, 'hidden_size', 'N/A')}")
            print(f"     隐藏层数量: {getattr(config, 'num_hidden_layers', 'N/A')}")
            print(f"     注意力头数: {getattr(config, 'num_attention_heads', 'N/A')}")
            print(f"     词汇表大小: {getattr(config, 'vocab_size', 'N/A')}")
            print(f"     最大位置编码: {getattr(config, 'max_position_embeddings', 'N/A')}")
            
            # 尝试加载tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
                print(f"   🔤 Tokenizer信息:")
                print(f"     Tokenizer类型: {type(tokenizer).__name__}")
                print(f"     词汇表大小: {tokenizer.vocab_size}")
                print(f"     Pad Token: {tokenizer.pad_token}")
                print(f"     EOS Token: {tokenizer.eos_token}")
                print(f"     BOS Token: {tokenizer.bos_token}")
            except Exception as e:
                print(f"   ⚠️  无法加载tokenizer: {str(e)[:50]}...")
            
            # 计算参数量 - 使用更安全的方式
            try:
                print(f"   🧠 正在计算参数量...")
                
                # 检查是否有权重文件
                weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
                if not weight_files:
                    print(f"   ⚠️  未找到权重文件，无法计算参数量")
                    return
                
                print(f"   📁 找到 {len(weight_files)} 个权重文件")
                
                # 使用更轻量的方式计算参数量
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16,
                    device_map='auto' if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True  # 减少内存使用
                )
                
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"   📈 参数量:")
                print(f"     总参数量: {total_params:,}")
                print(f"     可训练参数: {trainable_params:,}")
                print(f"     参数量(十亿): {total_params / 1e9:.2f}B")
                
                # 释放内存
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   ⚠️  无法计算参数量: {str(e)[:100]}...")
                # 即使无法加载模型，也继续处理，因为配置已经成功加载
            
        except Exception as e:
            print(f"   ❌ 无法分析模型信息: {str(e)[:100]}...")
            # 即使配置加载失败，也不要阻止模型被识别
        
        print()  # 添加空行分隔
    
    def get_model_size(self, model_path: Path):
        """获取模型大小 - 完全照抄model_downloader.py"""
        try:
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
            
            # 转换为可读格式
            if total_size >= 1024**3:
                return f"{total_size / 1024**3:.1f} GB"
            elif total_size >= 1024**2:
                return f"{total_size / 1024**2:.1f} MB"
            else:
                return f"{total_size / 1024:.1f} KB"
        except:
            return "Unknown"
    
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
            model_name: 模型名称或完整路径
        """
        # 如果model_name已经是完整路径，直接使用；否则拼接路径
        if os.path.isabs(model_name) or model_name.startswith('model/'):
            model_path = model_name
        else:
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
            
            # 强制使用GPU加载模型
            print("🚀 强制使用GPU加载模型...")
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
                
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"💾 GPU内存状态: 已用 {allocated:.2f}GB / 总计 {total:.1f}GB")
            
            # 强制使用GPU加载，不使用device_map="auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 强制移动到GPU
            print("🔄 将模型移动到GPU...")
            self.model = self.model.to(self.device)
            
            # 验证模型确实在GPU上
            if self.model.device.type != 'cuda':
                raise RuntimeError(f"模型未能成功移动到GPU，当前设备: {self.model.device}")
            
            print(f"✅ 模型已成功加载到GPU: {self.model.device}")
            
            # 显示GPU内存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU内存使用: {allocated:.2f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")
            
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
        # 处理模型路径，避免重复
        if self.selected_model_name.startswith('model/'):
            model_path = self.selected_model_name
        else:
            model_path = os.path.join(self.model_dir, self.selected_model_name)
        
        new_model_config = original_config.__class__.from_pretrained(model_path)
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
        print(f"   📊 新模型配置:")
        print(f"     隐藏层大小: {new_config['hidden_size']}")
        print(f"     层数: {new_config['num_hidden_layers']} (原: {self.original_layers_count})")
        print(f"     注意力头数: {new_config['num_attention_heads']}")
        print(f"     新增层数: {new_config['num_hidden_layers'] - self.original_layers_count}")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            print("   🧹 开始清理GPU内存...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   🧹 GPU内存清理完成，当前使用: {allocated:.2f}GB / 总计 {total:.1f}GB")
            
            # 如果内存使用率仍然很高，尝试释放更多缓存
            if allocated / total > 0.1:  # 降低阈值，更积极地清理
                print("   ⚠️  GPU内存使用率较高，尝试更激进的清理...")
                
                # 只清理缓存，不删除原模型
                print("   🧹 清理GPU缓存...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                print(f"   🧹 缓存清理完成，当前使用: {allocated:.2f}GB")
                
                # 如果还是很高，强制重置
                if allocated / total > 0.15:
                    print("   🚨 内存使用率仍然很高，强制重置CUDA缓存...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                    
                    # 尝试释放所有可能的缓存
                    for obj in gc.get_objects():
                        try:
                            if torch.is_tensor(obj) and obj.is_cuda:
                                del obj
                        except:
                            pass
                    
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"   🧹 强制清理完成，当前使用: {allocated:.2f}GB")
        
        print("   🔄 正在创建新模型配置...")
        print("   ⏳ 这可能需要几分钟时间，请耐心等待...")
        print("   💡 如果觉得太慢，可以按 Ctrl+C 中断，然后选择快速模式")
        
        # 140GB GPU专用优化策略 - 添加超时机制
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            free_memory = total_memory - allocated
            
            print(f"   💾 GPU内存状态: 已用 {allocated:.2f}GB / 总计 {total_memory:.1f}GB (可用 {free_memory:.2f}GB)")
            
            # 140GB GPU，使用超时机制
            print("   🚀 140GB GPU火力全开，使用GPU创建模型...")
            print("   ⏳ 模型创建可能需要一些时间，请耐心等待...")
            print("   ⏰ 设置60秒超时，如果超时将自动切换到快速模式")
            
            # 使用超时机制创建模型
            import signal
            import threading
            import time
            
            model_created = False
            new_model = None
            creation_error = None
            
            def create_model_with_timeout():
                nonlocal model_created, new_model, creation_error
                try:
                    # 设置CUDA优化
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    
                    print("   🔧 启用CUDA优化...")
                    print("   🔄 开始创建模型...")
                    
                    # 使用更快的创建方式
                    new_model = AutoModelForCausalLM.from_config(
                        new_model_config,
                        torch_dtype=torch.float16,  # 使用float16节省内存
                    )
                    model_created = True
                    print("   ✅ GPU模型创建成功")
                    
                except Exception as e:
                    creation_error = e
                    print(f"   ❌ GPU创建失败: {e}")
            
            # 启动模型创建线程
            creation_thread = threading.Thread(target=create_model_with_timeout)
            creation_thread.daemon = True
            creation_thread.start()
            
            # 等待模型创建，最多60秒
            start_time = time.time()
            timeout = 60  # 60秒超时
            
            while not model_created and time.time() - start_time < timeout:
                time.sleep(1)
                elapsed = int(time.time() - start_time)
                if elapsed % 10 == 0:  # 每10秒显示一次进度
                    print(f"   ⏳ 模型创建中... ({elapsed}s)")
            
            if not model_created:
                print(f"   ⏰ 模型创建超时 ({timeout}s)，切换到快速模式...")
                print("   🚀 使用快速模式创建模型（随机初始化新层）...")
                
                # 快速模式：在CPU上创建，然后移动到GPU
                try:
                    with torch.device('cpu'):
                        new_model = AutoModelForCausalLM.from_config(new_model_config)
                    print("   ✅ 快速模式模型创建成功")
                    
                    # 移动到GPU
                    if torch.cuda.is_available():
                        print("   🔄 将模型移动到GPU...")
                        new_model = new_model.to(self.device)
                        print("   ✅ 模型已移动到GPU")
                    
                    # 跳过权重复制，直接返回
                    print("   ⏭️  快速模式：跳过权重复制，新层将使用随机初始化")
                    self.model = new_model
                    print(f"模型扩展完成，新参数量: {self.model.num_parameters():,}")
                    return True
                    
                except Exception as e:
                    print(f"   ❌ 快速模式也失败: {e}")
                    return False
            else:
                print("   ✅ 模型创建成功")
                    
        else:
            print("   ❌ 未检测到GPU，无法使用GPU创建模型")
            return False
        
        # 确保模型在GPU上
        if torch.cuda.is_available():
            if new_model.device.type != 'cuda':
                print("   🔄 将模型移动到GPU...")
                try:
                    new_model = new_model.to(self.device)
                    print("   ✅ 模型已移动到GPU")
                except Exception as e:
                    print(f"   ❌ 模型移动到GPU失败: {e}")
                    return False
            else:
                print("   ✅ 模型已在GPU上")
        else:
            print("   ❌ 未检测到GPU，无法继续")
            return False
        
        # 复制原模型权重到新模型 - 添加超时机制
        print("复制原模型权重...")
        print(f"   📋 开始权重复制...")
        print(f"   📊 原模型参数量: {original_model.num_parameters():,}")
        print(f"   📊 新模型参数量: {new_model.num_parameters():,}")
        print(f"   📈 参数增长: {new_model.num_parameters() - original_model.num_parameters():,}")
        print(f"   ⏰ 设置120秒超时，如果超时将自动切换到快速模式")
        
        # 使用超时机制进行权重复制
        weights_copied = False
        copy_error = None
        
        def copy_weights_with_timeout():
            nonlocal weights_copied, copy_error
            try:
                self._copy_weights_preserving_knowledge(original_model, new_model)
                weights_copied = True
                print("   ✅ 权重复制完成")
            except Exception as e:
                copy_error = e
                print(f"   ❌ 权重复制失败: {e}")
        
        # 启动权重复制线程
        copy_thread = threading.Thread(target=copy_weights_with_timeout)
        copy_thread.daemon = True
        copy_thread.start()
        
        # 等待权重复制，最多120秒
        start_time = time.time()
        timeout = 120  # 120秒超时
        
        while not weights_copied and time.time() - start_time < timeout:
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            if elapsed % 15 == 0:  # 每15秒显示一次进度
                print(f"   ⏳ 权重复制中... ({elapsed}s)")
        
        if not weights_copied:
            print(f"   ⏰ 权重复制超时 ({timeout}s)，切换到快速模式...")
            print("   🚀 使用快速模式（新层将使用随机初始化）...")
            # 跳过权重复制，直接使用新模型
        else:
            print("   ✅ 权重复制成功完成")
        
        # 设置最终模型位置 - 140GB GPU专用
        print("设置最终模型位置...")
        if torch.cuda.is_available():
            print("   ✅ 140GB GPU，直接使用GPU模式")
            self.model = new_model
            
            # 确保模型在GPU上
            if self.model.device.type != 'cuda':
                print("   🔄 确保模型在GPU上...")
                self.model = self.model.to(self.device)
            
            print("   ✅ 模型已成功设置在GPU上")
        else:
            print("   ❌ 未检测到GPU，无法继续")
            return False
        
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
        total_layers = copy_layers
        for i in range(copy_layers):
            print(f"   🔄 复制第 {i+1}/{total_layers} 层...")
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
    
    def _freeze_layers(self, start_layer: int, end_layer: int, freeze: bool = True):
        """
        冻结或解冻指定范围的层
        
        Args:
            start_layer: 开始层索引
            end_layer: 结束层索引
            freeze: True为冻结，False为解冻
        """
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            print("❌ 模型结构不支持层冻结")
            return
        
        layers = self.model.model.layers
        action = "冻结" if freeze else "解冻"
        print(f"🔒 {action}第 {start_layer} 到 {end_layer} 层...")
        
        frozen_count = 0
        for i in range(start_layer, min(end_layer, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = not freeze
            frozen_count += 1
        
        print(f"✅ {action}了 {frozen_count} 层")
    
    def _get_trainable_parameters_count(self):
        """获取可训练参数数量"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        return trainable_params, total_params
    
    def _print_layer_status(self):
        """打印各层的训练状态"""
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            return
        
        layers = self.model.model.layers
        print("📊 各层训练状态:")
        
        for i in range(len(layers)):
            layer_params = list(layers[i].parameters())
            trainable = any(p.requires_grad for p in layer_params)
            status = "🟢 可训练" if trainable else "🔴 已冻结"
            print(f"   层 {i:2d}: {status}")
        
        trainable_params, total_params = self._get_trainable_parameters_count()
        print(f"📈 可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
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
        
        # 强制确保模型在GPU上
        if hasattr(trainer.model, 'device'):
            if trainer.model.device.type != 'cuda':
                print("⚠️  检测到模型不在GPU上，强制移动到GPU...")
                trainer.model = trainer.model.to(self.device)
                print(f"✅ 模型已移动到GPU: {trainer.model.device}")
        else:
            print("⚠️  无法检测模型设备，手动移动到GPU...")
            trainer.model = trainer.model.to(self.device)
            print(f"✅ 模型已移动到GPU: {trainer.model.device}")
        
        # 验证GPU使用
        print("🔍 验证GPU使用情况:")
        print(f"  模型设备: {trainer.model.device}")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU内存使用: {allocated:.2f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")
        
        return trainer
        
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
        训练扩展后的模型 - 使用分阶段训练策略
        
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
        
        # 执行分阶段训练
        return self._progressive_training(train_dataset, output_dir, epochs, batch_size)
    
    def _progressive_training(self, train_dataset, output_dir: str, epochs: int, batch_size: int):
        """
        分阶段渐进式训练策略
        
        阶段1: 冻结原有层，只训练新增层
        阶段2: 解冻部分顶层原有层，继续微调
        阶段3: 解冻全部层进行全量微调
        """
        print("🚀 开始分阶段渐进式训练策略")
        print("=" * 60)
        
        # 获取模型层数信息
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            print("❌ 模型结构不支持分阶段训练")
            return False
        
        total_layers = len(self.model.model.layers)
        original_layers = self.original_layers_count
        new_layers = total_layers - original_layers
        
        print(f"📊 模型层数信息:")
        print(f"  总层数: {total_layers}")
        print(f"  原有层数: {original_layers}")
        print(f"  新增层数: {new_layers}")
        
        # 阶段1: 只训练新增层
        print("\n🎯 阶段1: 冻结原有层，只训练新增层")
        print("-" * 40)
        
        # 冻结原有层 (0 到 original_layers-1)
        self._freeze_layers(0, original_layers, freeze=True)
        # 解冻新增层 (original_layers 到 total_layers-1)
        self._freeze_layers(original_layers, total_layers, freeze=False)
        
        self._print_layer_status()
        
        # 训练新增层
        stage1_output = f"{output_dir}/stage1_new_layers"
        print(f"\n🔄 开始训练新增层...")
        print(f"📁 输出目录: {stage1_output}")
        
        # 使用较小的学习率训练新增层
        stage1_epochs = max(1, epochs // 3)  # 阶段1使用1/3的epochs
        success = self._train_stage(train_dataset, stage1_output, stage1_epochs, batch_size, 
                                  learning_rate=1e-4, stage_name="新增层训练")
        
        if not success:
            print("❌ 阶段1训练失败")
            return False
        
        # 阶段2: 解冻部分顶层原有层
        print("\n🎯 阶段2: 解冻部分顶层原有层，继续微调")
        print("-" * 40)
        
        # 解冻最后1/3的原有层
        unfreeze_start = max(0, original_layers - original_layers // 3)
        self._freeze_layers(unfreeze_start, original_layers, freeze=False)
        
        self._print_layer_status()
        
        # 训练部分原有层 + 新增层
        stage2_output = f"{output_dir}/stage2_partial_unfreeze"
        print(f"\n🔄 开始训练部分原有层 + 新增层...")
        print(f"📁 输出目录: {stage2_output}")
        
        stage2_epochs = max(1, epochs // 3)  # 阶段2使用1/3的epochs
        success = self._train_stage(train_dataset, stage2_output, stage2_epochs, batch_size,
                                  learning_rate=5e-5, stage_name="部分层微调")
        
        if not success:
            print("❌ 阶段2训练失败")
            return False
        
        # 阶段3: 全量微调
        print("\n🎯 阶段3: 解冻全部层进行全量微调")
        print("-" * 40)
        
        # 解冻所有层
        self._freeze_layers(0, total_layers, freeze=False)
        
        self._print_layer_status()
        
        # 全量微调
        stage3_output = f"{output_dir}/stage3_full_finetune"
        print(f"\n🔄 开始全量微调...")
        print(f"📁 输出目录: {stage3_output}")
        
        stage3_epochs = max(1, epochs - stage1_epochs - stage2_epochs)  # 剩余epochs
        success = self._train_stage(train_dataset, stage3_output, stage3_epochs, batch_size,
                                  learning_rate=2e-5, stage_name="全量微调")
        
        if not success:
            print("❌ 阶段3训练失败")
            return False
        
        print("\n✅ 分阶段训练完成!")
        print(f"📁 最终模型保存在: {stage3_output}")
        
        # 加载最终模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(stage3_output, trust_remote_code=True)
            print("✅ 最终模型加载成功")
        except Exception as e:
            print(f"❌ 最终模型加载失败: {e}")
            return False
        
        return True
    
    def _train_stage(self, train_dataset, output_dir: str, epochs: int, batch_size: int, 
                    learning_rate: float, stage_name: str):
        """
        训练单个阶段
        
        Args:
            train_dataset: 训练数据集
            output_dir: 输出目录
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            stage_name: 阶段名称
        """
        print(f"🎯 {stage_name} - 训练参数:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Learning Rate: {learning_rate}")
        
        # 根据GPU大小设置梯度累积步数
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_memory >= 100:  # 140GB GPU
                gradient_accumulation_steps = 4
            elif total_memory >= 50:  # 50GB+ GPU
                gradient_accumulation_steps = 8
            else:
                gradient_accumulation_steps = 16
        else:
            gradient_accumulation_steps = 32
        
        print(f"  Gradient Accumulation Steps: {gradient_accumulation_steps}")
        
        try:
            # 创建训练器
            trainer = self._create_gradient_accumulation_trainer(
                train_dataset, output_dir, epochs, batch_size, gradient_accumulation_steps
            )
            
            # 设置自定义学习率
            trainer.learning_rate = learning_rate
            
            # 开始训练
            print(f"🚀 开始{stage_name}...")
            trainer.train()
            
            # 保存模型
            trainer.save_model()
            print(f"✅ {stage_name}完成，模型已保存到: {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ {stage_name}失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
    parser.add_argument("--model_dir", default="model", help="模型文件夹路径（相对于train_component目录）")
    parser.add_argument("--data_dir", default="data", help="数据文件夹路径")
    
    args = parser.parse_args()
    
    # 创建扩展器
    expander = ModelExpander(args.model_dir, args.data_dir)
    
    # 运行扩展管道
    expander.run_expansion_pipeline()

if __name__ == "__main__":
    main() 