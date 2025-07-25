#!/bin/bash

echo "🔄 GitHub到Gitee同步脚本"
echo "================================"

# 检查Git配置
if ! git remote get-url origin &> /dev/null; then
    echo "❌ 错误：当前目录不是Git仓库"
    exit 1
fi

# 获取当前远程仓库URL
CURRENT_REMOTE=$(git remote get-url origin)
echo "📍 当前远程仓库: $CURRENT_REMOTE"

# 检查是否是GitHub仓库
if [[ $CURRENT_REMOTE == *"github.com"* ]]; then
    echo "✅ 检测到GitHub仓库"
    
    # 提示用户输入Gitee仓库URL
    echo ""
    echo "📝 请输入您的Gitee仓库URL:"
    echo "   格式: https://gitee.com/用户名/仓库名.git"
    read -p "Gitee仓库URL: " GITEE_URL
    
    if [ -z "$GITEE_URL" ]; then
        echo "❌ 错误：未输入Gitee仓库URL"
        exit 1
    fi
    
    # 添加Gitee作为远程仓库
    echo "🔗 添加Gitee远程仓库..."
    git remote add gitee $GITEE_URL
    
    # 推送代码到Gitee
    echo "📤 推送代码到Gitee..."
    git push gitee master
    
    echo ""
    echo "✅ 同步完成！"
    echo "📍 GitHub仓库: $CURRENT_REMOTE"
    echo "📍 Gitee仓库: $GITEE_URL"
    echo ""
    echo "📋 后续同步命令："
    echo "   git push gitee master  # 推送到Gitee"
    echo "   git push origin master # 推送到GitHub"
    
elif [[ $CURRENT_REMOTE == *"gitee.com"* ]]; then
    echo "✅ 检测到Gitee仓库"
    echo "📍 当前已经是Gitee仓库，无需同步"
    
else
    echo "❓ 未知的远程仓库类型"
    echo "📍 当前仓库: $CURRENT_REMOTE"
fi

echo ""
echo "🎯 完成！" 