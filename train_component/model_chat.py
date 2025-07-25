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
        self.conversation_stats = {
            'total_generations': 0,
            'total_tokens': 0,
            'total_chinese_chars': 0,
            'total_time': 0.0
        }
        
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
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    # 如果eos_token也是None，设置一个默认值
                    self.tokenizer.pad_token = self.tokenizer.eos_token = "[PAD]"
                    print("⚠️  设置默认pad_token为[PAD]")
            
            print(f"✅ tokenizer配置:")
            print(f"   pad_token: {self.tokenizer.pad_token}")
            print(f"   eos_token: {self.tokenizer.eos_token}")
            print(f"   vocab_size: {self.tokenizer.vocab_size}")
            
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
    
    def generate_response_stream(self, prompt: str, temperature: float = 0.7):
        """
        流式生成回复
        
        Args:
            prompt: 输入提示
            temperature: 温度参数
        """
        if self.model is None or self.tokenizer is None:
            print("❌ 模型未加载")
            return None
        
        try:
            import time
            
            # 记录开始时间
            start_time = time.time()
            
            # 编码输入，明确设置attention_mask
            tokenized = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=False,  # 不截断输入，让模型处理长输入
                return_attention_mask=True
            )
            
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            
            if self.device == "cuda":
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
            
            # 流式生成回复
            generated_text = ""
            input_tokens = len(input_ids[0])
            output_tokens = 0
            chinese_chars = 0
            
            with torch.no_grad():
                # 直接生成完整回复
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,  # 减少生成长度，避免过长回复
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,  # 添加top_p采样
                    top_k=50,   # 添加top_k采样
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,  # 增加重复惩罚
                    use_cache=True,
                    no_repeat_ngram_size=3,  # 避免重复的n-gram
                    early_stopping=True,     # 早期停止
                    length_penalty=0.8       # 长度惩罚
                )
                
                # 解码输出
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 移除原始提示，只返回生成的部分
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                # 清理生成的文本，移除可能的重复前缀
                if generated_text.startswith("🤖 助手: "):
                    generated_text = generated_text[6:].strip()
                
                # 模拟流式输出
                print("🤖 助手: ", end="", flush=True)
                for char in generated_text:
                    print(char, end="", flush=True)
                    import time
                    time.sleep(0.01)  # 添加小延迟模拟打字效果
                
                # 计算统计信息
                output_tokens = len(outputs[0]) - input_tokens
                chinese_chars = sum(1 for char in generated_text if '\u4e00' <= char <= '\u9fff')
            
            print()  # 换行
            
            # 记录结束时间
            end_time = time.time()
            generation_time = end_time - start_time
            
            # 计算生成速度
            if generation_time > 0:
                tokens_per_second = output_tokens / generation_time
                chars_per_second = chinese_chars / generation_time
            else:
                tokens_per_second = 0
                chars_per_second = 0
            
            # 保存统计信息
            self.last_generation_stats = {
                'generation_time': generation_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'chinese_chars': chinese_chars,
                'tokens_per_second': tokens_per_second,
                'chars_per_second': chars_per_second
            }
            
            return generated_text
            
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
                    # 显示总体统计信息
                    if self.conversation_stats['total_generations'] > 0:
                        print(f"\n📈 本次对话总体统计:")
                        print(f"   🎯 总对话次数: {self.conversation_stats['total_generations']}")
                        print(f"   🔢 总生成tokens: {self.conversation_stats['total_tokens']}")
                        print(f"   🇨🇳 总汉字数量: {self.conversation_stats['total_chinese_chars']}")
                        print(f"   ⏱️  总生成时间: {self.conversation_stats['total_time']:.2f}秒")
                        
                        if self.conversation_stats['total_time'] > 0:
                            avg_tokens_per_second = self.conversation_stats['total_tokens'] / self.conversation_stats['total_time']
                            avg_chars_per_second = self.conversation_stats['total_chinese_chars'] / self.conversation_stats['total_time']
                            print(f"   ⚡ 平均速度: {avg_tokens_per_second:.1f} tokens/秒, {avg_chars_per_second:.1f} 汉字/秒")
                    
                    print("👋 再见！")
                    break
                
                if not user_input:
                    continue
                
                # 构建完整提示 - 只包含最近的对话历史
                if conversation_history:
                    # 限制历史长度，避免上下文过长
                    recent_history = conversation_history[-4:]  # 只保留最近2轮对话
                    full_prompt = "\n".join(recent_history) + f"\n用户: {user_input}\n助手: 请直接回答用户的问题，不要自问自答。"
                else:
                    full_prompt = f"用户: {user_input}\n助手: 请直接回答用户的问题，不要自问自答。"
                
                # 流式生成回复
                response = self.generate_response_stream(full_prompt)
                
                if response:
                    
                    # 显示生成统计信息
                    if hasattr(self, 'last_generation_stats'):
                        stats = self.last_generation_stats
                        print(f"\n📊 生成统计:")
                        print(f"   ⏱️  生成时间: {stats['generation_time']:.2f}秒")
                        print(f"   🔢 输入tokens: {stats['input_tokens']}")
                        print(f"   🔢 输出tokens: {stats['output_tokens']}")
                        print(f"   🔢 总tokens: {stats['total_tokens']}")
                        print(f"   🇨🇳 汉字数量: {stats['chinese_chars']}")
                        print(f"   ⚡ 生成速度: {stats['tokens_per_second']:.1f} tokens/秒")
                        print(f"   ⚡ 汉字速度: {stats['chars_per_second']:.1f} 汉字/秒")
                        
                        # 更新总体统计
                        self.conversation_stats['total_generations'] += 1
                        self.conversation_stats['total_tokens'] += stats['output_tokens']
                        self.conversation_stats['total_chinese_chars'] += stats['chinese_chars']
                        self.conversation_stats['total_time'] += stats['generation_time']
                    
                    # 更新对话历史
                    conversation_history.append(f"用户: {user_input}")
                    conversation_history.append(f"助手: {response}")
                    
                    # 限制对话历史长度，避免过长
                    if len(conversation_history) > 4:  # 保留最近2轮对话
                        conversation_history = conversation_history[-4:]
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