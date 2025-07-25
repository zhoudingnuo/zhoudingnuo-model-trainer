@echo off
chcp 65001 >nul
echo 🤖 模型工具启动器
echo ========================
echo.
echo 请选择要运行的工具:
echo 1. 模型下载器 - 从Hugging Face下载模型
echo 2. 模型对话器 - 与本地模型对话
echo 3. 模型扩展训练 - 训练模型
echo 4. 退出
echo.

set /p choice="请输入选择 (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🚀 启动模型下载器...
    python model_downloader.py
) else if "%choice%"=="2" (
    echo.
    echo 💬 启动模型对话器...
    python model_chat.py
) else if "%choice%"=="3" (
    echo.
    echo 🎯 启动模型扩展训练...
    python model_expansion.py
) else if "%choice%"=="4" (
    echo.
    echo 👋 再见！
    pause
    exit
) else (
    echo.
    echo ❌ 无效的选择，请重新运行脚本
    pause
    exit
)

echo.
echo 工具执行完成，按任意键退出...
pause >nul 