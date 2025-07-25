#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型对话器 - 调用本地模型进行对话
"""

import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import warnings
warnings.filterwarnings("ignore")

class ModelChat:
    def __init__(self, model_dir: str = "model"):
        """
        初始化模型对话器
        
        Args:
            model_dir: 模型目录
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def list_available_models(self):
        """列出可用的本地模型"""
        print("📚 可用的本地模型:")
        print("=" * 40)
        
        if not self.model_dir.exists():
            print("❌ 模型目录不存在")
            return []
        
        models = []
        for i, model_path in enumerate(self.model_dir.iterdir(), 1):
            if model_path.is_dir():
                # 检查是否有必要的文件
                config_file = model_path / "config.json"
                tokenizer_file = model_path / "tokenizer.json"
                
                if config_file.exists():
                    status = "✅ 完整模型" if tokenizer_file.exists() else "⚠️  部分模型"
                    print(f"{i:2d}. 📁 {model_path.name} ({status})")
                    models.append(str(model_path))
        
        return models
    
    def load_model(self, model_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"❌ 模型路径不存在: {model_path}")
            return False
        
        try:
            print(f"🔄 正在加载模型: {model_path.name}")
            
            # 加载tokenizer
            print("🔤 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), 
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ tokenizer加载完成")
            
            # 加载模型
            print("🧠 加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("✅ 模型加载完成")
            print(f"🎯 模型已加载到设备: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7):
        """
        生成回复
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
        """
        if self.model is None or self.tokenizer is None:
            print("❌ 模型未加载")
            return None
        
        try:
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 移除原始提示，只返回生成的部分
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ 生成回复失败: {e}")
            return None
    
    def chat_loop(self):
        """对话循环"""
        if self.model is None:
            print("❌ 请先加载模型")
            return
        
        print("💬 开始对话 (输入 'quit' 退出)")
        print("=" * 50)
        
        conversation_history = []
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 您: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not user_input:
                    continue
                
                # 构建完整提示
                if conversation_history:
                    full_prompt = "\n".join(conversation_history) + f"\n用户: {user_input}\n助手:"
                else:
                    full_prompt = f"用户: {user_input}\n助手:"
                
                # 生成回复
                print("🤖 助手: ", end="", flush=True)
                response = self.generate_response(full_prompt)
                
                if response:
                    print(response)
                    # 更新对话历史
                    conversation_history.append(f"用户: {user_input}")
                    conversation_history.append(f"助手: {response}")
                    
                    # 限制对话历史长度
                    if len(conversation_history) > 10:
                        conversation_history = conversation_history[-10:]
                else:
                    print("抱歉，我无法生成回复。")
                    
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 对话出错: {e}")
    
    def interactive_mode(self):
        """交互式模式"""
        print("🤖 模型对话器")
        print("=" * 50)
        
        while True:
            print("\n请选择操作:")
            print("1. 查看可用模型")
            print("2. 选择模型并开始对话")
            print("3. 重新加载当前模型")
            print("4. 退出")
            
            choice = input("\n请输入选择 (1-4): ").strip()
            
            if choice == "1":
                self.list_available_models()
                
            elif choice == "2":
                models = self.list_available_models()
                if models:
                    while True:
                        try:
                            model_choice = input(f"\n请选择模型 (1-{len(models)}): ").strip()
                            if not model_choice:
                                print("❌ 请输入选择")
                                continue
                            
                            choice_num = int(model_choice)
                            if 1 <= choice_num <= len(models):
                                selected_model = models[choice_num - 1]
                                print(f"✅ 已选择: {Path(selected_model).name}")
                                
                                # 加载模型
                                if self.load_model(selected_model):
                                    print("🚀 模型加载成功，开始对话...")
                                    self.chat_loop()
                                else:
                                    print("❌ 模型加载失败")
                                break
                            else:
                                print(f"❌ 请输入 1-{len(models)} 之间的数字")
                        except ValueError:
                            print("❌ 请输入有效的数字")
                else:
                    print("❌ 没有可用的模型")
                    
            elif choice == "3":
                if self.model is None:
                    print("❌ 当前没有加载的模型")
                else:
                    print("🔄 重新加载当前模型...")
                    # 这里可以添加重新加载逻辑
                    print("✅ 模型已重新加载")
                    
            elif choice == "4":
                print("👋 再见！")
                break
                
            else:
                print("❌ 无效的选择，请重新输入")

def main():
    """主函数"""
    chat = ModelChat()
    chat.interactive_mode()

if __name__ == "__main__":
    main() 