#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载器 - 从Hugging Face下载用户选择的模型
"""

import os
import sys
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

class ModelDownloader:
    def __init__(self, model_dir: str = "model"):
        """
        初始化模型下载器
        
        Args:
            model_dir: 模型保存目录
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def download_model(self, model_name: str):
        """
        下载模型
        
        Args:
            model_name: 模型名称
        """
        # 创建保存目录
        save_dir = self.model_dir / model_name.split('/')[-1]
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
                "download_time": str(datetime.now()),
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

def main():
    """主函数"""
    downloader = ModelDownloader()
    
    print("🚀 模型下载器")
    print("=" * 50)
    print("示例模型名称:")
    print("- microsoft/DialoGPT-medium")
    print("- Qwen/Qwen2.5-7B-Instruct")
    print("- THUDM/chatglm3-6b")
    print("- baichuan-inc/Baichuan2-7B-Chat")
    print()
    
    while True:
        print("请选择操作:")
        print("1. 下载模型")
        print("2. 查看已下载模型")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            model_name = input("请输入模型名称: ").strip()
            if model_name:
                downloader.download_model(model_name)
            else:
                print("❌ 模型名称不能为空")
                
        elif choice == "2":
            downloader.list_downloaded_models()
            
        elif choice == "3":
            print("👋 再见！")
            break
            
        else:
            print("❌ 无效的选择，请重新输入")

if __name__ == "__main__":
    main() 