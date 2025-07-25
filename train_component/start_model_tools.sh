#!/bin/bash

echo "🤖 模型工具启动器"
echo "========================"
echo ""
echo "请选择要运行的工具:"
echo "1. 模型下载器 - 支持Hugging Face和ModelScope"
echo "2. 模型对话器 - 与本地模型对话"
echo "3. 模型扩展训练 - 训练模型"
echo "4. 退出"
echo ""

read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 启动模型下载器..."
        echo "支持从Hugging Face和ModelScope下载模型"
        python3 model_downloader.py
        ;;
    2)
        echo ""
        echo "💬 启动模型对话器..."
        python3 model_chat.py
        ;;
    3)
        echo ""
        echo "🎯 启动模型扩展训练..."
        python3 model_expansion.py
        ;;
    4)
        echo ""
        echo "👋 再见！"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ 无效的选择，请重新运行脚本"
        exit 1
        ;;
esac

echo ""
echo "工具执行完成！" 