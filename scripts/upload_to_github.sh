#!/bin/bash

# GitHub Upload Script
# For automated code commit and push to GitHub

echo "🚀 Starting upload to GitHub..."

# Check if in Git repository
if [ ! -d ".git" ]; then
    echo "❌ Current directory is not a Git repository"
    echo "Please initialize Git repository first:"
    echo "  git init"
    echo "  git remote add origin https://github.com/your-username/your-repo.git"
    exit 1
fi

# Check remote repository
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "❌ Remote repository not found"
    echo "Please add remote repository:"
    echo "  git remote add origin https://github.com/your-username/your-repo.git"
    exit 1
fi

# Show current status
echo "📊 Current Git status:"
git status --short

# Add all files
echo "📁 Adding files to staging area..."
git add .

# Check if there are files to commit
if git diff --cached --quiet; then
    echo "✅ No files to commit"
    exit 0
fi

# Get commit message
if [ -z "$1" ]; then
    # Generate default commit message if not provided
    COMMIT_MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "📝 Using default commit message: $COMMIT_MSG"
else
    COMMIT_MSG="$1"
    echo "📝 Using custom commit message: $COMMIT_MSG"
fi

# Commit code
echo "💾 Committing code..."
git commit -m "$COMMIT_MSG"

# Push to remote repository
echo "📤 Pushing to GitHub..."
git push origin main

# Check push result
if [ $? -eq 0 ]; then
    echo "✅ Code upload successful!"
    echo "🌐 Repository URL: $(git remote get-url origin)"
else
    echo "❌ Push failed, please check network connection and permissions"
    exit 1
fi

echo ""
echo "🎯 Upload completed!"
echo "📋 Commit message: $COMMIT_MSG"
echo "🕐 Time: $(date)" 