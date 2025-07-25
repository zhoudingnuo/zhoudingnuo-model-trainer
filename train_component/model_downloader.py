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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
    
    def check_huggingface_access(self):
        """检查Hugging Face是否可访问"""
        try:
            import requests
            test_urls = [
                "https://huggingface.co",
                "https://hf-mirror.com",
                "https://huggingface.co.cn"
            ]
            
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=3)
                    if response.status_code == 200:
                        return True
                except:
                    continue
            
            return False
        except:
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
            working_url = None
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        print(f"✅ 网络连接正常: {url}")
                        network_ok = True
                        working_url = url
                        break
                except Exception as e:
                    print(f"❌ 连接失败: {url} - {e}")
                    continue
            
            if not network_ok:
                print("⚠️  所有Hugging Face镜像都无法访问")
                print("💡 建议:")
                print("   1. 使用ModelScope下载源")
                print("   2. 检查网络连接")
                print("   3. 配置VPN或代理")
                return False
            
            # 设置工作镜像
            if working_url:
                import os
                os.environ['HF_ENDPOINT'] = working_url
                os.environ['HF_HUB_URL'] = working_url
                print(f"🔧 使用镜像: {working_url}")
            
            # 配置镜像
            import os
            # 设置环境变量使用阿里镜像
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            os.environ['HF_HUB_URL'] = 'https://hf-mirror.com'
            
            # 设置huggingface_hub使用阿里镜像
            try:
                from huggingface_hub import set_http_backend
                set_http_backend("https://hf-mirror.com")
            except:
                pass
            
            # 检查模型是否需要登录
            print("🔍 检查模型访问权限...")
            try:
                api = HfApi()
                model_info = api.model_info(model_name)
                print(f"✅ 模型信息获取成功: {model_info.modelId}")
                print(f"   模型类型: {getattr(model_info, 'model_type', 'unknown')}")
                print(f"   是否私有: {getattr(model_info, 'private', False)}")
                
                if getattr(model_info, 'private', False):
                    print("⚠️  这是一个私有模型，需要登录才能下载")
                    print("💡 解决方案:")
                    print("   1. 使用Hugging Face CLI登录: huggingface-cli login")
                    print("   2. 使用ModelScope下载源")
                    print("   3. 寻找公开的替代模型")
                    return False
                    
            except Exception as e:
                print(f"⚠️  无法获取模型信息: {e}")
                print("💡 可能的原因:")
                print("   1. 模型不存在或名称错误")
                print("   2. 模型需要登录访问")
                print("   3. 网络连接问题")
                print("🔄 继续尝试下载...")
            
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
                        mirror='aliyun',  # 使用阿里镜像
                        use_auth_token=None
                    )
                    print("✅ tokenizer下载成功")
                    break
                except Exception as e:
                    error_msg = str(e)
                    if "401" in error_msg or "Unauthorized" in error_msg:
                        print("❌ 模型需要登录才能访问")
                        print("💡 解决方案:")
                        print("   1. 使用Hugging Face CLI登录:")
                        print("      pip install huggingface_hub")
                        print("      huggingface-cli login")
                        print("   2. 使用ModelScope下载源")
                        print("   3. 寻找公开的替代模型")
                        return False
                    elif "404" in error_msg or "Not Found" in error_msg:
                        print("❌ 模型不存在或名称错误")
                        print("💡 请检查模型名称是否正确")
                        return False
                    else:
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
                        mirror='aliyun',  # 使用阿里镜像
                        use_auth_token=None
                    )
                    print("✅ 模型下载成功")
                    break
                except Exception as e:
                    error_msg = str(e)
                    if "401" in error_msg or "Unauthorized" in error_msg:
                        print("❌ 模型需要登录才能访问")
                        print("💡 解决方案:")
                        print("   1. 使用Hugging Face CLI登录:")
                        print("      pip install huggingface_hub")
                        print("      huggingface-cli login")
                        print("   2. 使用ModelScope下载源")
                        print("   3. 寻找公开的替代模型")
                        return False
                    elif "404" in error_msg or "Not Found" in error_msg:
                        print("❌ 模型不存在或名称错误")
                        print("💡 请检查模型名称是否正确")
                        return False
                    else:
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
            # 首先检查git lfs是否安装
            print("🔍 检查git lfs...")
            result = subprocess.run("git lfs version", shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️  git lfs未安装，尝试安装...")
                try:
                    subprocess.run("curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash", shell=True)
                    subprocess.run("sudo apt-get install git-lfs", shell=True)
                    subprocess.run("git lfs install", shell=True)
                    print("✅ git lfs安装成功")
                except Exception as e:
                    print(f"❌ git lfs安装失败: {e}")
                    print("🔄 跳过git lfs，直接使用git...")
            
            # 尝试不同的镜像URL
            mirror_urls = [
                f"https://huggingface.co/{model_name}",
                f"https://hf-mirror.com/{model_name}",  # 阿里云镜像
                f"https://huggingface.co.cn/{model_name}",
                f"https://modelscope.cn/models/{model_name}"  # ModelScope镜像
            ]
            
            # 尝试使用git clone（不使用lfs）
            print("📥 使用git clone下载...")
            for url in mirror_urls:
                try:
                    print(f"🔧 尝试镜像: {url}")
                    cmd = f"git clone {url} {save_dir}"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ git clone下载成功")
                        return True
                    else:
                        print(f"❌ git clone失败: {result.stderr}")
                except Exception as e:
                    print(f"❌ git clone异常: {e}")
                    continue
            
            # 如果git失败，尝试使用wget
            print("📥 尝试使用wget下载...")
            for url in mirror_urls:
                try:
                    print(f"🔧 尝试wget镜像: {url}")
                    # 使用更简单的wget命令
                    cmd = f"wget -r -np -nH --cut-dirs=2 -R 'index.html*' {url}/tree/main -P {save_dir}"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ wget下载成功")
                        return True
                    else:
                        print(f"❌ wget失败: {result.stderr}")
                except Exception as e:
                    print(f"❌ wget异常: {e}")
                    continue
            
            # 最后尝试使用curl
            print("📥 尝试使用curl下载...")
            for url in mirror_urls:
                try:
                    print(f"🔧 尝试curl镜像: {url}")
                    cmd = f"curl -L -o {save_dir}/model.zip {url}/archive/main.zip"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ curl下载成功")
                        # 解压文件
                        subprocess.run(f"unzip {save_dir}/model.zip -d {save_dir}", shell=True)
                        return True
                    else:
                        print(f"❌ curl失败: {result.stderr}")
                except Exception as e:
                    print(f"❌ curl异常: {e}")
                    continue
            
            print("❌ 所有命令行下载方法都失败了")
            print("💡 建议:")
            print("   1. 检查网络连接")
            print("   2. 使用ModelScope下载源")
            print("   3. 手动下载模型文件")
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
            
            # 验证模型名称格式
            if '/' not in model_name:
                print(f"❌ ModelScope模型名称格式错误: {model_name}")
                print("💡 ModelScope模型名称格式应为: namespace/name")
                print("   例如: YIRONGCHEN/SoulChat2.0-Yi-1.5-9B")
                return False
            
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
        # 创建保存目录 - 使用模型名称的最后一部分作为目录名
        if '/' in model_name:
            save_dir_name = model_name.split('/')[-1]
        else:
            save_dir_name = model_name
        
        save_dir = self.model_dir / save_dir_name
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
            
            # 检测网络环境
            print("🔍 检测网络环境...")
            hf_accessible = self.check_huggingface_access()
            
            if hf_accessible:
                print("🌐 Hugging Face可访问，优先尝试...")
                success = self.download_from_huggingface(model_name, save_dir)
                
                if not success:
                    print("🔄 Hugging Face失败，尝试ModelScope...")
                    if not self.check_modelscope_installed():
                        if not self.install_modelscope():
                            print("❌ ModelScope安装失败")
                            return None
                    success = self.download_from_modelscope(model_name, save_dir)
            else:
                print("🌐 Hugging Face不可访问，直接使用ModelScope...")
                if not self.check_modelscope_installed():
                    if not self.install_modelscope():
                        print("❌ ModelScope安装失败")
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
        for i, model_path in enumerate(self.model_dir.iterdir(), 1):
            if model_path.is_dir():
                info_file = model_path / "model_info.json"
                if info_file.exists():
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
                    except:
                        print(f"{i}. 📁 {model_path.name} (信息文件损坏)")
                        models.append(str(model_path))
                else:
                    print(f"{i}. 📁 {model_path.name} (无信息文件)")
                    # 尝试显示模型详细信息
                    self.show_model_details(model_path)
                    models.append(str(model_path))
        
        return models
    
    def show_model_details(self, model_path: Path):
        """显示模型的详细信息"""
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
            
            # 计算参数量
            try:
                print(f"   🧠 正在计算参数量...")
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16,
                    device_map='auto' if torch.cuda.is_available() else None,
                    trust_remote_code=True
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
                print(f"   ⚠️  无法计算参数量: {str(e)[:50]}...")
            
        except Exception as e:
            print(f"   ❌ 无法分析模型信息: {str(e)[:50]}...")
        
        print()  # 添加空行分隔
    
    def get_model_size(self, model_path: Path):
        """获取模型大小"""
        try:
            total_size = 0
            file_count = 0
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            # 转换为可读格式
            if total_size < 1024:
                return f"{total_size} B"
            elif total_size < 1024 * 1024:
                return f"{total_size / 1024:.1f} KB"
            elif total_size < 1024 * 1024 * 1024:
                return f"{total_size / (1024 * 1024):.1f} MB"
            else:
                return f"{total_size / (1024 * 1024 * 1024):.1f} GB"
        except:
            return "未知"
    
    def delete_model(self, model_name: str):
        """删除模型"""
        model_path = self.model_dir / model_name
        
        if not model_path.exists():
            print(f"❌ 模型不存在: {model_name}")
            return False
        
        if not model_path.is_dir():
            print(f"❌ 不是有效的模型目录: {model_name}")
            return False
        
        # 显示模型信息
        info_file = model_path / "model_info.json"
        if info_file.exists():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    info = json.load(f)
                print(f"🗑️  准备删除模型:")
                print(f"   名称: {info.get('name', model_name)}")
                print(f"   下载源: {info.get('source', 'Unknown')}")
                print(f"   下载时间: {info.get('download_time', 'Unknown')}")
                print(f"   大小: {self.get_model_size(model_path)}")
            except:
                print(f"🗑️  准备删除模型: {model_name}")
        else:
            print(f"🗑️  准备删除模型: {model_name}")
        
        # 确认删除
        confirm = input("⚠️  确定要删除这个模型吗？(输入 'DELETE' 确认): ").strip()
        if confirm != "DELETE":
            print("❌ 删除已取消")
            return False
        
        try:
            import shutil
            shutil.rmtree(model_path)
            print(f"✅ 模型删除成功: {model_name}")
            return True
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    
    def delete_model_by_index(self, index: int):
        """根据索引删除模型"""
        models = []
        for model_path in self.model_dir.iterdir():
            if model_path.is_dir():
                models.append(model_path.name)
        
        if index < 1 or index > len(models):
            print(f"❌ 无效的索引: {index}")
            return False
        
        model_name = models[index - 1]
        return self.delete_model(model_name)
    
    def clean_invalid_models(self):
        """一键清理无效模型"""
        print("🧹 开始扫描无效模型...")
        print("=" * 50)
        
        if not self.model_dir.exists():
            print("❌ 模型目录不存在")
            return
        
        invalid_models = []
        valid_models = []
        total_size_freed = 0
        
        for model_path in self.model_dir.iterdir():
            if not model_path.is_dir():
                continue
                
            print(f"🔍 检查模型: {model_path.name}")
            
            # 检查是否为训练输出目录
            if model_path.name in ['trained', 'output', 'checkpoints', 'logs', 'temp', 'tmp']:
                print(f"   ⏭️  跳过训练目录: {model_path.name}")
                continue
            
            # 检查是否有必要的模型文件
            config_file = model_path / "config.json"
            tokenizer_file = model_path / "tokenizer.json"
            model_info_file = model_path / "model_info.json"
            
            is_valid = True
            issues = []
            
            # 检查配置文件
            if not config_file.exists():
                is_valid = False
                issues.append("缺少config.json")
            
            # 检查tokenizer文件
            if not tokenizer_file.exists():
                issues.append("缺少tokenizer.json")
            
            # 检查模型信息文件
            if not model_info_file.exists():
                issues.append("缺少model_info.json")
            
            # 检查是否有模型权重文件
            weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
            if not weight_files:
                is_valid = False
                issues.append("缺少模型权重文件")
            
            # 检查目录是否为空或只有隐藏文件
            visible_files = [f for f in model_path.iterdir() if not f.name.startswith('.')]
            if not visible_files:
                is_valid = False
                issues.append("目录为空")
            
            if is_valid:
                valid_models.append(model_path)
                print(f"   ✅ 有效模型")
            else:
                invalid_models.append((model_path, issues))
                print(f"   ❌ 无效模型: {', '.join(issues)}")
        
        print("\n📊 扫描结果:")
        print(f"   有效模型: {len(valid_models)}")
        print(f"   无效模型: {len(invalid_models)}")
        
        if not invalid_models:
            print("✅ 没有发现无效模型")
            return
        
        print("\n🗑️  发现的无效模型:")
        for i, (model_path, issues) in enumerate(invalid_models, 1):
            size = self.get_model_size(model_path)
            print(f"   {i}. {model_path.name} ({size}) - {', '.join(issues)}")
        
        # 询问用户是否清理
        print(f"\n⚠️  即将删除 {len(invalid_models)} 个无效模型")
        confirm = input("确认删除吗？(y/N): ").strip().lower()
        
        if confirm in ['y', 'yes', '是']:
            print("\n🧹 开始清理无效模型...")
            
            for model_path, issues in invalid_models:
                try:
                    # 计算删除前的大小
                    size_before = self.get_model_size(model_path)
                    
                    # 删除目录
                    import shutil
                    shutil.rmtree(model_path)
                    
                    print(f"   ✅ 已删除: {model_path.name} ({size_before})")
                    
                    # 尝试解析大小以计算总释放空间
                    try:
                        if 'GB' in size_before:
                            size_gb = float(size_before.replace(' GB', ''))
                            total_size_freed += size_gb
                        elif 'MB' in size_before:
                            size_mb = float(size_before.replace(' MB', ''))
                            total_size_freed += size_mb / 1024
                    except:
                        pass
                        
                except Exception as e:
                    print(f"   ❌ 删除失败: {model_path.name} - {e}")
            
            print(f"\n✅ 清理完成!")
            if total_size_freed > 0:
                print(f"📊 释放空间: {total_size_freed:.2f} GB")
        else:
            print("❌ 取消清理")

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
    print("- 如果Hugging Face无法访问，程序会自动选择ModelScope")
    print()
    print("示例模型名称:")
    print("ModelScope（推荐）:")
    print("- YIRONGCHEN/SoulChat2.0-Yi-1.5-9B")
    print("- qwen/Qwen2.5-7B-Instruct")
    print("- THUDM/chatglm3-6b")
    print("- YIRONGCHEN/SoulChat2.0-Llama-3.1-8B")
    print()
    print("Hugging Face:")
    print("- microsoft/DialoGPT-medium")
    print("- Qwen/Qwen2.5-7B-Instruct")
    print("- THUDM/chatglm3-6b")
    print("- baichuan-inc/Baichuan2-7B-Chat")
    print()
    
    while True:
        print("\n请选择操作:")
        print("1. 下载模型 (自动选择源)")
        print("2. 从Hugging Face下载")
        print("3. 从ModelScope下载")
        print("4. 查看已下载模型")
        print("5. 查看模型详细信息")
        print("6. 删除模型")
        print("7. 一键清理无效模型")
        print("8. 退出")
        
        choice = input("\n请输入选择 (1-8): ").strip()
        
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
            print("\n📊 查看模型详细信息")
            print("=" * 30)
            models = downloader.list_downloaded_models()
            
            if not models:
                print("❌ 没有可查看的模型")
                continue
            
            try:
                index = int(input("请输入要查看详细信息的模型索引: ").strip())
                if 1 <= index <= len(models):
                    model_path = Path(models[index - 1])
                    print(f"\n🔍 正在分析模型: {model_path.name}")
                    print("=" * 50)
                    downloader.show_model_details(model_path)
                else:
                    print("❌ 无效的索引")
            except ValueError:
                print("❌ 请输入有效的数字")
            
        elif choice == "6":
            print("\n🗑️  删除模型")
            print("=" * 30)
            models = downloader.list_downloaded_models()
            
            if not models:
                print("❌ 没有可删除的模型")
                continue
            
            print("\n选择删除方式:")
            print("1. 按索引删除")
            print("2. 按名称删除")
            print("3. 返回主菜单")
            
            delete_choice = input("\n请输入选择 (1-3): ").strip()
            
            if delete_choice == "1":
                try:
                    index = int(input("请输入模型索引: ").strip())
                    downloader.delete_model_by_index(index)
                except ValueError:
                    print("❌ 请输入有效的数字")
                    
            elif delete_choice == "2":
                model_name = input("请输入模型名称: ").strip()
                if model_name:
                    downloader.delete_model(model_name)
                else:
                    print("❌ 模型名称不能为空")
                    
            elif delete_choice == "3":
                continue
            else:
                print("❌ 无效的选择")
            
        elif choice == "7":
            downloader.clean_invalid_models()
            
        elif choice == "8":
            print("👋 再见！")
            break
            
        else:
            print("❌ 无效的选择，请重新输入")

if __name__ == "__main__":
    main() 