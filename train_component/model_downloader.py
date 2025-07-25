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
            import requests
            from huggingface_hub import HfApi
            
            # 测试网络连接
            print("🔍 测试网络连接...")
            test_urls = [
                "https://huggingface.co",
                "https://hf-mirror.com",  # 国内镜像
                "https://huggingface.co.cn"  # 另一个镜像
            ]
            
            network_ok = False
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        print(f"✅ 网络连接正常: {url}")
                        network_ok = True
                        break
                except Exception as e:
                    print(f"❌ 连接失败: {url} - {e}")
                    continue
            
            if not network_ok:
                print("⚠️  所有网络连接都失败")
                print("💡 建议使用ModelScope下载源")
                return False
            
            # 配置镜像
            import os
            # 设置环境变量使用国内镜像
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            os.environ['HF_HUB_URL'] = 'https://hf-mirror.com'
            
            # 设置huggingface_hub使用镜像
            try:
                from huggingface_hub import set_http_backend
                set_http_backend("https://hf-mirror.com")
            except:
                pass
            
            print("🔤 下载tokenizer...")
            # 尝试不同的代理配置
            proxy_configs = [
                None,  # 无代理
                {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'},  # 常见代理端口
                {'http': 'http://127.0.0.1:1080', 'https': 'http://127.0.0.1:1080'},  # 另一个常见端口
            ]
            
            for proxies in proxy_configs:
                try:
                    print(f"🔧 尝试代理配置: {proxies}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        cache_dir=save_dir,
                        local_files_only=False,
                        resume_download=True,
                        proxies=proxies,
                        mirror='tuna',  # 使用清华镜像
                        use_auth_token=None
                    )
                    print("✅ tokenizer下载成功")
                    break
                except Exception as e:
                    print(f"❌ 代理配置失败: {e}")
                    continue
            else:
                raise Exception("所有代理配置都失败了")
            tokenizer.save_pretrained(save_dir)
            print("✅ tokenizer下载完成")
            
            # 下载模型
            print("🧠 下载模型...")
            # 使用相同的代理配置
            for proxies in proxy_configs:
                try:
                    print(f"🔧 尝试代理配置下载模型: {proxies}")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        cache_dir=save_dir,
                        torch_dtype=torch.float16,
                        device_map="auto" if torch.cuda.is_available() else None,
                        local_files_only=False,
                        resume_download=True,
                        proxies=proxies,
                        mirror='tuna',  # 使用清华镜像
                        use_auth_token=None
                    )
                    print("✅ 模型下载成功")
                    break
                except Exception as e:
                    print(f"❌ 模型下载代理配置失败: {e}")
                    continue
            else:
                raise Exception("所有模型下载代理配置都失败了")
            model.save_pretrained(save_dir)
            print("✅ 模型下载完成")
            
            return True
            
        except requests.exceptions.ConnectionError as e:
            print(f"❌ 网络连接失败: {e}")
            print("💡 建议:")
            print("   1. 检查网络连接")
            print("   2. 使用ModelScope下载源")
            print("   3. 配置代理或VPN")
            return False
        except requests.exceptions.Timeout as e:
            print(f"❌ 网络超时: {e}")
            print("💡 建议使用ModelScope下载源")
            return False
        except Exception as e:
            print(f"❌ Hugging Face下载失败: {e}")
            print("🔄 尝试使用命令行下载...")
            return self.download_with_cli(model_name, save_dir)
    
    def download_with_cli(self, model_name: str, save_dir: Path):
        """使用命令行工具下载模型"""
        print("🔧 使用命令行下载...")
        
        try:
            # 尝试使用git lfs
            print("📥 使用git lfs下载...")
            # 尝试不同的镜像URL
            mirror_urls = [
                f"https://huggingface.co/{model_name}",
                f"https://hf-mirror.com/{model_name}",
                f"https://huggingface.co.cn/{model_name}"
            ]
            
            for url in mirror_urls:
                try:
                    print(f"🔧 尝试镜像: {url}")
                    cmd = f"git lfs install && git clone {url} {save_dir}"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ 命令行下载成功")
                        return True
                    else:
                        print(f"❌ 镜像失败: {result.stderr}")
                except Exception as e:
                    print(f"❌ 镜像异常: {e}")
                    continue
            
            # 如果git lfs失败，尝试使用wget
            print("📥 尝试使用wget下载...")
            for url in mirror_urls:
                try:
                    print(f"🔧 尝试wget镜像: {url}")
                    cmd = f"wget -r -np -nH --cut-dirs=2 -R 'index.html*' {url}/tree/main -P {save_dir}"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ wget下载成功")
                        return True
                    else:
                        print(f"❌ wget镜像失败: {result.stderr}")
                except Exception as e:
                    print(f"❌ wget镜像异常: {e}")
                    continue
            
            print("❌ 所有命令行下载方法都失败了")
            return False
            
        except Exception as e:
            print(f"❌ 命令行下载失败: {e}")
            return False
    
    def download_from_modelscope(self, model_name: str, save_dir: Path):
        """从ModelScope下载模型"""
        print("🏢 从ModelScope下载...")
        
        try:
            # 使用modelscope的Python API下载
            from modelscope import snapshot_download
            
            print(f"开始下载模型: {model_name}")
            print(f"保存到: {save_dir}")
            
            # 使用snapshot_download API
            downloaded_path = snapshot_download(
                model_id=model_name,
                cache_dir=str(save_dir),
                local_dir=str(save_dir)
            )
            
            print(f"✅ ModelScope下载完成: {downloaded_path}")
            return True
                
        except ImportError:
            print("❌ ModelScope未安装，尝试安装...")
            if self.install_modelscope():
                # 重新尝试下载
                try:
                    from modelscope import snapshot_download
                    
                    print(f"重新开始下载模型: {model_name}")
                    downloaded_path = snapshot_download(
                        model_id=model_name,
                        cache_dir=str(save_dir),
                        local_dir=str(save_dir)
                    )
                    
                    print(f"✅ ModelScope下载完成: {downloaded_path}")
                    return True
                except Exception as e:
                    print(f"❌ ModelScope下载失败: {e}")
                    return False
            else:
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
            
            # 优先尝试ModelScope（国内网络更稳定）
            print("1️⃣ 优先尝试ModelScope下载...")
            if not self.check_modelscope_installed():
                if not self.install_modelscope():
                    print("❌ ModelScope安装失败")
                    return None
            success = self.download_from_modelscope(model_name, save_dir)
            
            if not success:
                # 如果ModelScope失败，尝试Hugging Face
                print("2️⃣ ModelScope失败，尝试Hugging Face...")
                success = self.download_from_huggingface(model_name, save_dir)
        
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
    print("- ModelScope: 阿里云模型社区（推荐，国内网络稳定）")
    print("- Hugging Face: 全球最大的模型社区")
    print()
    print("💡 建议:")
    print("- 国内用户优先使用ModelScope")
    print("- 如果网络不稳定，选择ModelScope下载源")
    print()
    print("示例模型名称:")
    print("ModelScope（推荐）:")
    print("- YIRONGCHEN/SoulChat2.0-Yi-1.5-9B")
    print("- qwen/Qwen2.5-7B-Instruct")
    print("- THUDM/chatglm3-6b")
    print()
    print("Hugging Face:")
    print("- microsoft/DialoGPT-medium")
    print("- Qwen/Qwen2.5-7B-Instruct")
    print("- THUDM/chatglm3-6b")
    print("- baichuan-inc/Baichuan2-7B-Chat")
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