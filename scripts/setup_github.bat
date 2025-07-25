@echo off
chcp 65001 >nul
echo 🚀 GitHub Repository Initialization Script
echo.

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed, please install Git first
    echo Download: https://git-scm.com/downloads
    pause
    exit /b 1
)

echo ✅ Git is installed: 
git --version

REM Check if Git repository is already initialized
if exist ".git" (
    echo ✅ Git repository already exists
    goto :check_remote
) else (
    echo 📁 Initializing Git repository...
    git init
    if errorlevel 1 (
        echo ❌ Git initialization failed
        pause
        exit /b 1
    )
    echo ✅ Git repository initialized successfully
)

:check_remote
REM Check remote repository
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo 📝 Please set GitHub repository URL
    set /p REPO_URL="Enter GitHub repository URL (e.g., https://github.com/username/repo.git): "
    if "%REPO_URL%"=="" (
        echo ❌ No repository URL entered
        pause
        exit /b 1
    )
    git remote add origin "%REPO_URL%"
    echo ✅ Remote repository set successfully
) else (
    echo ✅ Remote repository already configured
    echo Current remote repository: 
    git remote get-url origin
)

REM Create .gitignore file
if not exist ".gitignore" (
    echo 📄 Creating .gitignore file...
    (
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo *$py.class
        echo *.so
        echo .Python
        echo build/
        echo develop-eggs/
        echo dist/
        echo downloads/
        echo eggs/
        echo .eggs/
        echo lib/
        echo lib64/
        echo parts/
        echo sdist/
        echo var/
        echo wheels/
        echo *.egg-info/
        echo .installed.cfg
        echo *.egg
        echo MANIFEST
        echo.
        echo # PyTorch
        echo *.pth
        echo *.pt
        echo *.ckpt
        echo.
        echo # Model files
        echo expanded_models/
        echo distilled_models/
        echo *.safetensors
        echo.
        echo # Log files
        echo logs/
        echo *.log
        echo.
        echo # Temporary files
        echo temp/
        echo tmp/
        echo.
        echo # System files
        echo .DS_Store
        echo Thumbs.db
        echo.
        echo # IDE
        echo .vscode/
        echo .idea/
        echo *.swp
        echo *.swo
        echo.
        echo # Data files
        echo data/*.jsonl
        echo data/*.json
        echo data/*.txt
        echo.
        echo # Environment files
        echo .env
        echo .venv/
        echo venv/
        echo env/
    ) > .gitignore
    echo ✅ .gitignore file created successfully
)

REM Set execution permissions
echo 🔧 Setting script execution permissions...
if exist "upload_to_github.sh" (
    echo chmod +x upload_to_github.sh
)
if exist "update_from_github.sh" (
    echo chmod +x update_from_github.sh
)
if exist "setup_vast_ai.sh" (
    echo chmod +x setup_vast_ai.sh
)

echo.
echo 🎯 GitHub repository setup completed!
echo.
echo 📋 Usage instructions:
echo   1. Upload code: ./upload_to_github.sh [commit message]
echo   2. Update code: ./update_from_github.sh
echo   3. Check status: git status
echo   4. View logs: git log --oneline
echo.
echo 💡 Tips:
echo   - First use: ./upload_to_github.sh "Initial commit"
echo   - Remember to backup important files regularly
echo   - Don't commit large model files to GitHub
echo.
pause 