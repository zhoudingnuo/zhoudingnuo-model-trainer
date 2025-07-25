#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载器 - 从Hugging Face下载用户选择的模型
"""

import os
import sys
import json
import requests
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from tqdm import tqdm

class ModelDownloader:
    def __init__(self, model_dir: str = "model"):
        """
        初始化模型下载器
        
        Args:
            model_dir: 模型保存目录
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # 预定义的模型列表
        self.popular_models = {
            "1": {
                "name": "Qwen/Qwen2.5-7B-Instruct",
                "description": "Qwen2.5 7B 指令微调模型",
                "size": "约14GB"
            },
            "2": {
                "name": "microsoft/DialoGPT-medium",
                "description": "微软DialoGPT中等模型",
                "size": "约1.5GB"
            },
            "3": {
                "name": "THUDM/chatglm3-6b",
                "description": "清华ChatGLM3 6B模型",
                "size": "约12GB"
            },
            "4": {
                "name": "baichuan-inc/Baichuan2-7B-Chat",
                "description": "百川2 7B对话模型",
                "size": "约14GB"
            },
            "5": {
                "name": "internlm/internlm2-chat-7b",
                "description": "InternLM2 7B对话模型",
                "size": "约14GB"
            },
            "6": {
                "name": "custom",
                "description": "自定义模型",
                "size": "未知"
            }
        }
    
    def show_model_list(self):
        """显示可用的模型列表"""
        print("🤖 可用的模型列表:")
        print("=" * 60)
        for key, model_info in self.popular_models.items():
            print(f"{key}. {model_info['name']}")
            print(f"   描述: {model_info['description']}")
            print(f"   大小: {model_info['size']}")
            print()
    
    def get_model_info(self, model_name: str):
        """获取模型信息"""
        try:
            # 尝试获取模型信息
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            
            # 计算模型大小
            param_size = sum(p.numel() for p in model.parameters())
            model_size_mb = param_size * 4 / (1024 * 1024)  # 假设float32
            
            return {
                "name": model_name,
                "vocab_size": tokenizer.vocab_size,
                "model_size_mb": model_size_mb,
                "available": True
            }
        except Exception as e:
            return {
                "name": model_name,
                "error": str(e),
                "available": False
            }
    
    def download_model(self, model_name: str, save_dir: str = None):
        """
        下载模型
        
        Args:
            model_name: 模型名称
            save_dir: 保存目录，如果为None则使用默认目录
        """
        if save_dir is None:
            save_dir = self.model_dir / model_name.split('/')[-1]
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 开始下载模型: {model_name}")
        print(f"📁 保存目录: {save_dir}")
        print()
        
        try:
            # 下载tokenizer
            print("🔤 下载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir=save_dir
            )
            tokenizer.save_pretrained(save_dir)
            print("✅ tokenizer下载完成")
            
            # 下载模型
            print("🧠 下载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=save_dir,
                torch_dtype=torch.float16,  # 使用半精度节省空间
                device_map="auto" if torch.cuda.is_available() else None
            )
            model.save_pretrained(save_dir)
            print("✅ 模型下载完成")
            
            # 保存模型信息
            model_info = {
                "name": model_name,
                "local_path": str(save_dir),
                "download_time": str(torch.datetime.now()),
                "model_type": "causal_lm"
            }
            
            with open(save_dir / "model_info.json", "w", encoding="utf-8") as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            print(f"🎉 模型下载完成！保存在: {save_dir}")
            return str(save_dir)
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return None
    
    def list_downloaded_models(self):
        """列出已下载的模型"""
        print("📚 已下载的模型:")
        print("=" * 40)
        
        if not self.model_dir.exists():
            print("❌ 模型目录不存在")
            return []
        
        models = []
        for model_path in self.model_dir.iterdir():
            if model_path.is_dir():
                info_file = model_path / "model_info.json"
                if info_file.exists():
                    try:
                        with open(info_file, "r", encoding="utf-8") as f:
                            info = json.load(f)
                        print(f"📁 {model_path.name}")
                        print(f"   原始名称: {info.get('name', 'Unknown')}")
                        print(f"   下载时间: {info.get('download_time', 'Unknown')}")
                        models.append(str(model_path))
                    except:
                        print(f"📁 {model_path.name} (信息文件损坏)")
                else:
                    print(f"📁 {model_path.name} (无信息文件)")
        
        return models
    
    def interactive_download(self):
        """交互式下载模型"""
        print("🚀 模型下载器")
        print("=" * 50)
        
        while True:
            print("\n请选择操作:")
            print("1. 查看可用模型列表")
            print("2. 下载预定义模型")
            print("3. 下载自定义模型")
            print("4. 查看已下载模型")
            print("5. 退出")
            
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == "1":
                self.show_model_list()
                
            elif choice == "2":
                self.show_model_list()
                model_choice = input("\n请选择模型编号: ").strip()
                
                if model_choice in self.popular_models:
                    model_name = self.popular_models[model_choice]["name"]
                    if model_name == "custom":
                        model_name = input("请输入自定义模型名称: ").strip()
                    
                    if model_name:
                        self.download_model(model_name)
                else:
                    print("❌ 无效的选择")
                    
            elif choice == "3":
                model_name = input("请输入模型名称 (例如: microsoft/DialoGPT-medium): ").strip()
                if model_name:
                    self.download_model(model_name)
                else:
                    print("❌ 模型名称不能为空")
                    
            elif choice == "4":
                self.list_downloaded_models()
                
            elif choice == "5":
                print("👋 再见！")
                break
                
            else:
                print("❌ 无效的选择，请重新输入")

def main():
    """主函数"""
    downloader = ModelDownloader()
    downloader.interactive_download()

if __name__ == "__main__":
    main() 