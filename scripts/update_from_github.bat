@echo off
chcp 65001 >nul
echo 🔄 Starting update from GitHub...

REM Check if in Git repository
if not exist ".git" (
    echo ❌ Current directory is not a Git repository
    echo Please clone GitHub repository first:
    echo   git clone https://github.com/your-username/your-repo.git
    pause
    exit /b 1
)

REM Check remote repository
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo ❌ Remote repository not found
    echo Please add remote repository:
    echo   git remote add origin https://github.com/your-username/your-repo.git
    pause
    exit /b 1
)

REM Show current branch
for /f "tokens=*" %%i in ('git branch --show-current') do set CURRENT_BRANCH=%%i
echo 🌿 Current branch: %CURRENT_BRANCH%

REM Save current work
echo 💾 Saving current work...
git stash

REM Pull latest code
echo 📥 Pulling latest code...
git fetch origin

REM Check if there are updates
for /f "tokens=*" %%i in ('git rev-parse HEAD') do set LOCAL_COMMIT=%%i
for /f "tokens=*" %%i in ('git rev-parse origin/main') do set REMOTE_COMMIT=%%i

if "%LOCAL_COMMIT%"=="%REMOTE_COMMIT%" (
    echo ✅ Code is already up to date
    git stash pop
    pause
    exit /b 0
)

REM Merge latest code
echo 🔀 Merging latest code...
git pull origin main

REM Check merge result
if errorlevel 1 (
    echo ❌ Merge failed, please resolve conflicts manually
    echo 💡 Tips:
    echo   1. View conflict files: git status
    echo   2. After resolving conflicts: git add .
    echo   3. Complete merge: git commit
    pause
    exit /b 1
) else (
    echo ✅ Code update successful!
    
    REM Restore previous work
    git stash list | findstr /r "." >nul
    if not errorlevel 1 (
        echo 🔄 Restoring previous work...
        git stash pop
    )
    
    REM Show update information
    echo.
    echo 📋 Update information:
    echo   Local commit: %LOCAL_COMMIT%
    echo   Remote commit: %REMOTE_COMMIT%
    echo   Update time: %date% %time%
    
    REM Show recent commits
    echo.
    echo 📝 Recent commits:
    git log --oneline -5
)

echo.
echo 🎯 Update completed!
pause 