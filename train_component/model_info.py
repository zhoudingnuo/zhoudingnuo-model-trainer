#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型信息检测脚本
用于检测本地模型文件夹中的模型配置信息
"""

import os
import json
from transformers import AutoConfig, AutoTokenizer
import torch

class ModelInfoDetector:
    def __init__(self, model_dir="D:\\Model"):
        self.model_dir = model_dir
    
    def list_models(self):
        """列出模型文件夹中的所有模型"""
        models = []
        
        if not os.path.exists(self.model_dir):
            print(f"模型目录不存在: {self.model_dir}")
            return models
        
        for item in os.listdir(self.model_dir):
            item_path = os.path.join(self.model_dir, item)
            if os.path.isdir(item_path):
                # 检查是否是模型目录
                if self._is_model_directory(item_path):
                    models.append(item)
        
        return models
    
    def _is_model_directory(self, path):
        """检查目录是否包含模型文件"""
        # 检查常见的模型文件
        model_files = [
            'config.json',
            'tokenizer.json', 
            'tokenizer_config.json',
            'pytorch_model.bin',
            'model.safetensors'
        ]
        
        # 检查直接文件
        for file in model_files:
            if os.path.exists(os.path.join(path, file)):
                return True
        
        # 检查snapshots子目录（Hugging Face Hub格式）
        snapshots_dir = os.path.join(path, 'snapshots')
        if os.path.exists(snapshots_dir):
            for snapshot in os.listdir(snapshots_dir):
                snapshot_path = os.path.join(snapshots_dir, snapshot)
                if os.path.isdir(snapshot_path):
                    for file in model_files:
                        if os.path.exists(os.path.join(snapshot_path, file)):
                            return True
        
        return False
    
    def get_model_path(self, model_name):
        """获取模型的实际路径"""
        model_dir = os.path.join(self.model_dir, model_name)
        
        # 检查直接路径
        if os.path.exists(os.path.join(model_dir, 'config.json')):
            return model_dir
        
        # 检查snapshots子目录
        snapshots_dir = os.path.join(model_dir, 'snapshots')
        if os.path.exists(snapshots_dir):
            for snapshot in os.listdir(snapshots_dir):
                snapshot_path = os.path.join(snapshots_dir, snapshot)
                if os.path.exists(os.path.join(snapshot_path, 'config.json')):
                    return snapshot_path
        
        return None
    
    def detect_model_info(self, model_name):
        """检测指定模型的配置信息"""
        model_path = self.get_model_path(model_name)
        
        if not model_path:
            print(f"无法找到模型: {model_name}")
            return None
        
        try:
            # 加载配置
            config = AutoConfig.from_pretrained(model_path)
            
            # 提取关键信息
            info = {
                'model_name': model_name,
                'model_path': model_path,
                'model_type': getattr(config, 'model_type', 'unknown'),
                'hidden_size': getattr(config, 'hidden_size', None),
                'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
                'num_attention_heads': getattr(config, 'num_attention_heads', None),
                'vocab_size': getattr(config, 'vocab_size', None),
                'intermediate_size': getattr(config, 'intermediate_size', None),
                'max_position_embeddings': getattr(config, 'max_position_embeddings', None),
                'rope_theta': getattr(config, 'rope_theta', None),
                'rms_norm_eps': getattr(config, 'rms_norm_eps', None),
                'use_cache': getattr(config, 'use_cache', None),
                'pad_token_id': getattr(config, 'pad_token_id', None),
                'bos_token_id': getattr(config, 'bos_token_id', None),
                'eos_token_id': getattr(config, 'eos_token_id', None),
                'tie_word_embeddings': getattr(config, 'tie_word_embeddings', None),
                'torch_dtype': getattr(config, 'torch_dtype', None),
                'transformers_version': getattr(config, 'transformers_version', None)
            }
            
            # 计算参数量
            try:
                # 尝试加载模型来计算参数量
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    device_map='auto',
                    trust_remote_code=True
                )
                
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                info['total_parameters'] = total_params
                info['trainable_parameters'] = trainable_params
                info['parameters_in_billions'] = total_params / 1e9
                
                # 释放内存
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"计算参数量时出错: {e}")
                info['total_parameters'] = None
                info['trainable_parameters'] = None
                info['parameters_in_billions'] = None
            
            return info
            
        except Exception as e:
            print(f"检测模型信息时出错: {e}")
            return None
    
    def print_model_info(self, info):
        """打印模型信息"""
        if not info:
            return
        
        print("\n" + "="*60)
        print(f"模型名称: {info['model_name']}")
        print(f"模型路径: {info['model_path']}")
        print(f"模型类型: {info['model_type']}")
        print("-"*60)
        print("核心配置:")
        print(f"  hidden_size: {info['hidden_size']}")
        print(f"  num_hidden_layers: {info['num_hidden_layers']}")
        print(f"  num_attention_heads: {info['num_attention_heads']}")
        print(f"  vocab_size: {info['vocab_size']}")
        print(f"  intermediate_size: {info['intermediate_size']}")
        print(f"  max_position_embeddings: {info['max_position_embeddings']}")
        print("-"*60)
        print("其他配置:")
        print(f"  rope_theta: {info['rope_theta']}")
        print(f"  rms_norm_eps: {info['rms_norm_eps']}")
        print(f"  use_cache: {info['use_cache']}")
        print(f"  pad_token_id: {info['pad_token_id']}")
        print(f"  bos_token_id: {info['bos_token_id']}")
        print(f"  eos_token_id: {info['eos_token_id']}")
        print(f"  tie_word_embeddings: {info['tie_word_embeddings']}")
        print(f"  torch_dtype: {info['torch_dtype']}")
        print(f"  transformers_version: {info['transformers_version']}")
        
        if info['total_parameters']:
            print("-"*60)
            print("参数量:")
            print(f"  总参数量: {info['total_parameters']:,}")
            print(f"  可训练参数: {info['trainable_parameters']:,}")
            print(f"  参数量(十亿): {info['parameters_in_billions']:.2f}B")
        
        print("="*60)
    
    def run_detection(self):
        """运行检测流程"""
        print("正在扫描模型目录...")
        models = self.list_models()
        
        if not models:
            print(f"在 {self.model_dir} 中没有找到模型")
            return
        
        print(f"找到 {len(models)} 个模型:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        print("\n请选择要检测的模型 (输入序号，或输入 'all' 检测所有模型):")
        choice = input().strip()
        
        if choice.lower() == 'all':
            # 检测所有模型
            for model in models:
                print(f"\n正在检测模型: {model}")
                info = self.detect_model_info(model)
                self.print_model_info(info)
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(models):
                    model = models[index]
                    print(f"\n正在检测模型: {model}")
                    info = self.detect_model_info(model)
                    self.print_model_info(info)
                else:
                    print("无效的选择")
            except ValueError:
                print("请输入有效的数字")

def main():
    detector = ModelInfoDetector()
    detector.run_detection()

if __name__ == "__main__":
    main() 