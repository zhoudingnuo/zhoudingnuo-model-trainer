# 🚀 GitHub 使用指南

## 📋 快速开始

### 1. 初始化GitHub仓库

**Windows用户：**
```bash
# 运行初始化脚本
setup_github.bat
```

**Linux/Mac用户：**
```bash
# 手动初始化
git init
git remote add origin https://github.com/your-username/your-repo.git
```

### 2. 上传代码到GitHub

**Windows用户：**
```bash
# 使用默认提交信息
upload_to_github.bat

# 使用自定义提交信息
upload_to_github.bat "添加新功能：模型扩展训练"
```

**Linux/Mac用户：**
```bash
# 使用默认提交信息
./upload_to_github.sh

# 使用自定义提交信息
./upload_to_github.sh "添加新功能：模型扩展训练"
```

### 3. 从GitHub更新代码

**Windows用户：**
```bash
update_from_github.bat
```

**Linux/Mac用户：**
```bash
./update_from_github.sh
```

## 🔧 详细使用说明

### 初始化步骤

1. **安装Git**
   - Windows: 下载并安装 [Git for Windows](https://git-scm.com/downloads)
   - Linux: `sudo apt install git` (Ubuntu/Debian)
   - Mac: `brew install git` (使用Homebrew)

2. **配置Git用户信息**
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

3. **创建GitHub仓库**
   - 访问 [GitHub](https://github.com)
   - 点击 "New repository"
   - 填写仓库名称和描述
   - 不要初始化README文件（我们会手动添加）

4. **运行初始化脚本**
   ```bash
   # Windows
   setup_github.bat
   
   # Linux/Mac
   chmod +x setup_github.sh
   ./setup_github.sh
   ```

### 日常使用流程

#### 上传代码
```bash
# 1. 修改代码
# 2. 测试代码
# 3. 上传到GitHub

# Windows
upload_to_github.bat "描述你的修改"

# Linux/Mac
./upload_to_github.sh "描述你的修改"
```

#### 更新代码
```bash
# 1. 从GitHub拉取最新代码
# 2. 解决冲突（如果有）
# 3. 继续开发

# Windows
update_from_github.bat

# Linux/Mac
./update_from_github.sh
```

## 📁 文件说明

### 脚本文件
- `setup_github.bat` / `setup_github.sh` - GitHub仓库初始化
- `upload_to_github.bat` / `upload_to_github.sh` - 上传代码到GitHub
- `update_from_github.bat` / `update_from_github.sh` - 从GitHub更新代码

### 配置文件
- `.gitignore` - 排除不需要提交的文件
- `requirements.txt` - Python依赖包列表
- `README.md` - 项目说明文档

## 🛠️ 高级功能

### 分支管理
```bash
# 创建新分支
git checkout -b feature/new-feature

# 切换分支
git checkout main

# 合并分支
git merge feature/new-feature

# 删除分支
git branch -d feature/new-feature
```

### 版本回退
```bash
# 查看提交历史
git log --oneline

# 回退到指定版本
git reset --hard <commit-hash>

# 回退到上一个版本
git reset --hard HEAD~1
```

### 冲突解决
```bash
# 查看冲突文件
git status

# 编辑冲突文件，手动解决冲突
# 在冲突标记处选择保留的代码

# 添加解决后的文件
git add .

# 完成合并
git commit
```

## 🔒 安全注意事项

### 不要提交的文件
- 模型文件 (`.safetensors`, `.pth`, `.pt`)
- 训练数据 (`.jsonl`, `.json`, `.txt`)
- 日志文件 (`.log`)
- 临时文件 (`temp/`, `tmp/`)
- 环境文件 (`.env`, `venv/`)

### 敏感信息保护
- 不要在代码中硬编码API密钥
- 使用环境变量存储敏感信息
- 定期检查提交历史中的敏感信息

## 📊 最佳实践

### 提交信息规范
```bash
# 好的提交信息
upload_to_github.bat "feat: 添加模型扩展功能"
upload_to_github.bat "fix: 修复内存泄漏问题"
upload_to_github.bat "docs: 更新README文档"
upload_to_github.bat "style: 格式化代码"
upload_to_github.bat "refactor: 重构训练逻辑"

# 避免的提交信息
upload_to_github.bat "update"
upload_to_github.bat "fix bug"
upload_to_github.bat "."
```

### 工作流程
1. **开发前**：`update_from_github.bat` 获取最新代码
2. **开发中**：定期提交小改动
3. **开发后**：`upload_to_github.bat` 上传代码
4. **测试**：确保代码能正常运行

### 备份策略
- 重要文件定期备份到本地
- 使用Git标签标记重要版本
- 考虑使用GitHub Releases发布稳定版本

## 🚨 故障排除

### 常见问题

**推送失败**
```bash
# 检查网络连接
ping github.com

# 检查认证
git config --list | grep user

# 重新认证
git config --global credential.helper store
```

**合并冲突**
```bash
# 查看冲突文件
git status

# 手动编辑冲突文件
# 删除冲突标记，保留需要的代码

# 添加解决后的文件
git add .
git commit
```

**权限问题**
```bash
# 检查远程仓库URL
git remote -v

# 更新远程仓库URL
git remote set-url origin https://github.com/username/repo.git
```

### 错误信息

**"fatal: not a git repository"**
- 运行 `setup_github.bat` 初始化仓库

**"fatal: remote origin already exists"**
- 删除现有远程仓库：`git remote remove origin`
- 重新添加：`git remote add origin <url>`

**"fatal: refusing to merge unrelated histories"**
- 使用：`git pull origin main --allow-unrelated-histories`

## 📈 进阶技巧

### 自动化脚本
```bash
# 创建自动化部署脚本
echo "git add . && git commit -m 'Auto update' && git push" > auto_update.sh
chmod +x auto_update.sh
```

### 钩子脚本
```bash
# 在.git/hooks/pre-commit中添加测试
#!/bin/bash
python -m pytest tests/
```

### 持续集成
- 使用GitHub Actions自动测试
- 配置自动部署到vast.ai
- 设置代码质量检查

## 📞 支持

- **GitHub文档**: https://docs.github.com/
- **Git文档**: https://git-scm.com/doc
- **问题反馈**: 创建GitHub Issue

---

**💡 提示**：
- 定期备份重要文件
- 使用有意义的提交信息
- 及时解决合并冲突
- 保护敏感信息 