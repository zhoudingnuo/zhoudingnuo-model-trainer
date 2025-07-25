#!/bin/bash

# GitHub Update Script
# For pulling latest code from GitHub

echo "🔄 Starting update from GitHub..."

# Check if in Git repository
if [ ! -d ".git" ]; then
    echo "❌ Current directory is not a Git repository"
    echo "Please clone GitHub repository first:"
    echo "  git clone https://github.com/your-username/your-repo.git"
    exit 1
fi

# Check remote repository
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "❌ Remote repository not found"
    echo "Please add remote repository:"
    echo "  git remote add origin https://github.com/your-username/your-repo.git"
    exit 1
fi

# Show current branch
echo "🌿 Current branch: $(git branch --show-current)"

# Save current work
echo "💾 Saving current work..."
git stash

# Pull latest code
echo "📥 Pulling latest code..."
git fetch origin

# Check if there are updates
LOCAL_COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git rev-parse origin/main)

if [ "$LOCAL_COMMIT" = "$REMOTE_COMMIT" ]; then
    echo "✅ Code is already up to date"
    git stash pop
    exit 0
fi

# Merge latest code
echo "🔀 Merging latest code..."
git pull origin main

# Check merge result
if [ $? -eq 0 ]; then
    echo "✅ Code update successful!"
    
    # Restore previous work
    if git stash list | grep -q .; then
        echo "🔄 Restoring previous work..."
        git stash pop
    fi
    
    # Show update information
    echo ""
    echo "📋 Update information:"
    echo "  Local commit: $LOCAL_COMMIT"
    echo "  Remote commit: $REMOTE_COMMIT"
    echo "  Update time: $(date)"
    
    # Show recent commits
    echo ""
    echo "📝 Recent commits:"
    git log --oneline -5
    
else
    echo "❌ Merge failed, please resolve conflicts manually"
    echo "💡 Tips:"
    echo "  1. View conflict files: git status"
    echo "  2. After resolving conflicts: git add ."
    echo "  3. Complete merge: git commit"
    exit 1
fi

echo ""
echo "🎯 Update completed!" 