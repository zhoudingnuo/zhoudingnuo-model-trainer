@echo off
chcp 65001 >nul
echo 🚀 Starting upload to GitHub...

REM Check if in Git repository
if not exist ".git" (
    echo ❌ Current directory is not a Git repository
    echo Please run setup_github.bat to initialize Git repository
    pause
    exit /b 1
)

REM Check remote repository
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo ❌ Remote repository not found
    echo Please run setup_github.bat to configure remote repository
    pause
    exit /b 1
)

REM Show current status
echo 📊 Current Git status:
git status --short

REM Add all files
echo 📁 Adding files to staging area...
git add .

REM Check if there are files to commit
git diff --cached --quiet
if errorlevel 1 (
    echo ✅ Files need to be committed
) else (
    echo ✅ No files to commit
    pause
    exit /b 0
)

REM Get commit message
if "%1"=="" (
    REM Generate default commit message if not provided
    for /f "tokens=1-6 delims=/: " %%a in ('echo %date% %time%') do (
        set COMMIT_MSG=Update: %%a-%%b-%%c %%d:%%e:%%f
    )
    echo 📝 Using default commit message: %COMMIT_MSG%
) else (
    set COMMIT_MSG=%*
    echo 📝 Using custom commit message: %COMMIT_MSG%
)

REM Commit code
echo 💾 Committing code...
git commit -m "%COMMIT_MSG%"

REM Push to remote repository
echo 📤 Pushing to GitHub...
git push origin main

REM Check push result
if errorlevel 1 (
    echo ❌ Push failed, please check network connection and permissions
    pause
    exit /b 1
) else (
    echo ✅ Code upload successful!
    echo 🌐 Repository URL: 
    git remote get-url origin
)

echo.
echo 🎯 Upload completed!
echo 📋 Commit message: %COMMIT_MSG%
echo 🕐 Time: %date% %time%
pause 