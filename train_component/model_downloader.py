#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载器 - 支持从Hugging Face和ModelScope下载模型
"""

import os
import sys
import json
import subprocess
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
    
    def check_modelscope_installed(self):
        """检查ModelScope是否已安装"""
        try:
            import modelscope
            return True
        except ImportError:
            return False
    
    def install_modelscope(self):
        """安装ModelScope"""
        print("📦 正在安装ModelScope...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
            print("✅ ModelScope安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ ModelScope安装失败: {e}")
            return False
    
    def download_from_huggingface(self, model_name: str, save_dir: Path):
        """从Hugging Face下载模型"""
        print("🌐 从Hugging Face下载...")
        
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
            
            return True
            
        except Exception as e:
            print(f"❌ Hugging Face下载失败: {e}")
            return False
    
    def download_from_modelscope(self, model_name: str, save_dir: Path):
        """从ModelScope下载模型"""
        print("🏢 从ModelScope下载...")
        
        try:
            # 使用modelscope命令行下载
            cmd = [
                sys.executable, "-m", "modelscope", "download",
                "--model", model_name,
                "--local_dir", str(save_dir)
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ ModelScope下载完成")
                return True
            else:
                print(f"❌ ModelScope下载失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ ModelScope下载异常: {e}")
            return False
    
    def download_model(self, model_name: str, source: str = "auto"):
        """
        下载模型
        
        Args:
            model_name: 模型名称
            source: 下载源 ("huggingface", "modelscope", "auto")
        """
        # 创建保存目录
        save_dir = self.model_dir / model_name.split('/')[-1]
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 开始下载模型: {model_name}")
        print(f"📁 保存目录: {save_dir}")
        print(f"🌐 下载源: {source}")
        print()
        
        success = False
        
        if source == "huggingface":
            success = self.download_from_huggingface(model_name, save_dir)
        elif source == "modelscope":
            if not self.check_modelscope_installed():
                if not self.install_modelscope():
                    return None
            success = self.download_from_modelscope(model_name, save_dir)
        elif source == "auto":
            # 自动选择下载源
            print("🔄 自动选择下载源...")
            
            # 先尝试Hugging Face
            print("1️⃣ 尝试从Hugging Face下载...")
            success = self.download_from_huggingface(model_name, save_dir)
            
            if not success:
                # 如果Hugging Face失败，尝试ModelScope
                print("2️⃣ Hugging Face失败，尝试ModelScope...")
                if not self.check_modelscope_installed():
                    if not self.install_modelscope():
                        return None
                success = self.download_from_modelscope(model_name, save_dir)
        
        if success:
            # 保存模型信息
            model_info = {
                "name": model_name,
                "source": source,
                "local_path": str(save_dir),
                "download_time": str(datetime.now()),
                "model_type": "causal_lm"
            }
            
            with open(save_dir / "model_info.json", "w", encoding="utf-8") as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            print(f"🎉 模型下载完成！保存在: {save_dir}")
            return str(save_dir)
        else:
            print(f"❌ 所有下载源都失败了")
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
                        print(f"   下载源: {info.get('source', 'Unknown')}")
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
    print("支持的下载源:")
    print("- Hugging Face: 全球最大的模型社区")
    print("- ModelScope: 阿里云模型社区")
    print()
    print("示例模型名称:")
    print("Hugging Face:")
    print("- microsoft/DialoGPT-medium")
    print("- Qwen/Qwen2.5-7B-Instruct")
    print("- THUDM/chatglm3-6b")
    print("- baichuan-inc/Baichuan2-7B-Chat")
    print()
    print("ModelScope:")
    print("- YIRONGCHEN/SoulChat2.0-Yi-1.5-9B")
    print("- qwen/Qwen2.5-7B-Instruct")
    print("- THUDM/chatglm3-6b")
    print()
    
    while True:
        print("请选择操作:")
        print("1. 下载模型 (自动选择源)")
        print("2. 从Hugging Face下载")
        print("3. 从ModelScope下载")
        print("4. 查看已下载模型")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            model_name = input("请输入模型名称: ").strip()
            if model_name:
                downloader.download_model(model_name, "auto")
            else:
                print("❌ 模型名称不能为空")
                
        elif choice == "2":
            model_name = input("请输入Hugging Face模型名称: ").strip()
            if model_name:
                downloader.download_model(model_name, "huggingface")
            else:
                print("❌ 模型名称不能为空")
                
        elif choice == "3":
            model_name = input("请输入ModelScope模型名称: ").strip()
            if model_name:
                downloader.download_model(model_name, "modelscope")
            else:
                print("❌ 模型名称不能为空")
                
        elif choice == "4":
            downloader.list_downloaded_models()
            
        elif choice == "5":
            print("👋 再见！")
            break
            
        else:
            print("❌ 无效的选择，请重新输入")

if __name__ == "__main__":
    main() 