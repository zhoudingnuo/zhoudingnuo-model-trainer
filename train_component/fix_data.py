#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据修复脚本
"""

import json
import os
import re

def fix_jsonl_data(input_file, output_file):
    """修复JSONL数据文件"""
    print(f"修复数据文件: {input_file} -> {output_file}")
    
    fixed_lines = []
    line_count = 0
    fixed_count = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            line_count += 1
            line = line.strip()
            
            if not line:
                continue
                
            try:
                # 尝试修复常见的格式问题
                fixed_line = fix_line(line)
                if fixed_line:
                    fixed_lines.append(fixed_line)
                    fixed_count += 1
                    print(f"修复第{line_num}行: {fixed_line[:100]}...")
                else:
                    print(f"跳过第{line_num}行: 无法修复")
                    
            except Exception as e:
                print(f"处理第{line_num}行时出错: {e}")
                continue
    
    # 写入修复后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    print(f"修复完成: {fixed_count}/{line_count} 行数据")
    return fixed_count > 0

def fix_line(line):
    """修复单行数据"""
    # 移除乱码字符
    line = re.sub(r'[^\x00-\x7F\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+', '', line)
    
    # 尝试解析JSON
    try:
        # 如果已经是有效的JSON
        data = json.loads(line)
        if 'text' in data:
            return line
        else:
            # 添加text字段
            data['text'] = str(data)
            return json.dumps(data, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    
    # 尝试修复常见的格式问题
    if line.startswith('"') and not line.startswith('{"'):
        # 添加开头的{
        line = '{' + line
    
    if not line.endswith('}'):
        # 添加结尾的}
        line = line + '}'
    
    # 尝试添加text字段
    if '"text"' not in line:
        # 提取文件名或其他内容作为text
        text_content = extract_text_content(line)
        if text_content:
            # 在最后一个}之前插入text字段
            line = line[:-1] + f', "text": "{text_content}"' + line[-1]
    
    # 再次尝试解析
    try:
        data = json.loads(line)
        return json.dumps(data, ensure_ascii=False)
    except json.JSONDecodeError:
        return None

def extract_text_content(line):
    """从行中提取文本内容"""
    # 尝试提取文件名
    filename_match = re.search(r'"([^"]*\.pdf)"', line)
    if filename_match:
        return f"文档: {filename_match.group(1)}"
    
    # 尝试提取其他内容
    content_match = re.search(r'[一-龯]+', line)
    if content_match:
        return content_match.group(0)
    
    return "未知内容"

def create_sample_data(output_file):
    """创建示例数据文件"""
    print(f"创建示例数据文件: {output_file}")
    
    sample_data = [
        {"text": "这是一个示例训练文本，用于测试模型扩展功能。"},
        {"text": "人工智能技术正在快速发展，深度学习模型在各个领域都有广泛应用。"},
        {"text": "自然语言处理是人工智能的重要分支，包括文本分类、机器翻译等任务。"},
        {"text": "大语言模型如GPT、BERT等在NLP任务中表现出色。"},
        {"text": "模型训练需要大量的数据和计算资源，GPU加速训练是常用的方法。"},
        {"text": "增量学习允许模型在已有知识基础上继续学习新知识。"},
        {"text": "知识蒸馏可以将大模型的知识传递给小模型，提高小模型的性能。"},
        {"text": "模型压缩技术可以减少模型大小，提高推理速度。"},
        {"text": "注意力机制是Transformer模型的核心组件，能够捕捉序列中的长距离依赖。"},
        {"text": "预训练模型通过在大规模语料上训练，学习通用的语言表示。"}
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"创建了 {len(sample_data)} 条示例数据")

def main():
    """主函数"""
    input_file = "data/batch_training_data.jsonl"
    fixed_file = "data/fixed_training_data.jsonl"
    sample_file = "data/sample_training_data.jsonl"
    
    print("=== 数据修复工具 ===")
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        print("创建示例数据文件...")
        create_sample_data(sample_file)
        return
    
    # 尝试修复数据
    print("尝试修复现有数据...")
    if fix_jsonl_data(input_file, fixed_file):
        print(f"数据已修复并保存到: {fixed_file}")
    else:
        print("修复失败，创建示例数据...")
        create_sample_data(sample_file)
    
    # 显示修复后的数据
    print("\n修复后的数据预览:")
    try:
        with open(fixed_file if os.path.exists(fixed_file) else sample_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i <= 3:
                    print(f"第{i}行: {line.strip()}")
                else:
                    break
    except Exception as e:
        print(f"读取修复后的数据时出错: {e}")

if __name__ == "__main__":
    main() 