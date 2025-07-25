# 脚本工具集 (Scripts)

这个目录包含了项目中的各种自动化脚本工具和相关的使用指南文档。

## 脚本分类

### 🐧 Linux/Mac 脚本 (.sh)

#### 1. `setup_vast_ai.sh`
- **用途**: vast.ai 云GPU平台自动部署脚本
- **功能**: 
  - 自动安装Python依赖
  - 配置CUDA环境
  - 设置训练环境
- **使用方法**: 
  ```bash
  chmod +x setup_vast_ai.sh
  ./setup_vast_ai.sh
  ```

#### 2. `upload_to_github.sh`
- **用途**: 上传代码到GitHub仓库
- **功能**: 
  - 自动提交代码更改
  - 推送到远程仓库
  - 处理Git认证
- **使用方法**: 
  ```bash
  chmod +x upload_to_github.sh
  ./upload_to_github.sh
  ```

#### 3. `update_from_github.sh`
- **用途**: 从GitHub仓库更新代码
- **功能**: 
  - 拉取最新代码
  - 合并远程更改
  - 处理冲突
- **使用方法**: 
  ```bash
  chmod +x update_from_github.sh
  ./update_from_github.sh
  ```

### 🪟 Windows 脚本 (.bat)

#### 1. `setup_github.bat`
- **用途**: Windows环境下的GitHub仓库初始化
- **功能**: 
  - 配置Git用户信息
  - 初始化本地仓库
  - 设置远程仓库连接
- **使用方法**: 
  ```cmd
  setup_github.bat
  ```

#### 2. `upload_to_github.bat`
- **用途**: Windows环境下上传代码到GitHub
- **功能**: 
  - 自动提交代码更改
  - 推送到远程仓库
  - 处理Windows环境下的Git操作
- **使用方法**: 
  ```cmd
  upload_to_github.bat
  ```

#### 3. `update_from_github.bat`
- **用途**: Windows环境下从GitHub更新代码
- **功能**: 
  - 拉取最新代码
  - 合并远程更改
  - 处理Windows环境下的Git操作
- **使用方法**: 
  ```cmd
  update_from_github.bat
  ```

## 使用建议

### 首次使用
1. 如果是Linux/Mac环境，先运行 `setup_vast_ai.sh` 配置环境
2. 如果是Windows环境，先运行 `setup_github.bat` 配置Git

### 日常开发
1. 修改代码后，使用对应的上传脚本提交更改
2. 需要同步最新代码时，使用对应的更新脚本

### 注意事项
- 确保有适当的文件执行权限（Linux/Mac）
- Windows脚本需要在命令提示符或PowerShell中运行
- 使用Git脚本前确保已配置Git用户信息
- 建议在运行脚本前备份重要文件

## 脚本依赖

- **Git**: 所有GitHub相关脚本都需要安装Git
- **Python**: vast.ai脚本需要Python环境
- **Bash**: Linux/Mac脚本需要Bash shell
- **PowerShell/CMD**: Windows脚本需要Windows命令行环境

## 📚 相关文档

### `GITHUB_GUIDE.md`
- **用途**: GitHub使用详细指南
- **内容**: 
  - GitHub仓库创建和管理
  - 代码上传和更新流程
  - 常见问题解决方案
- **适用场景**: 需要管理GitHub仓库时参考

### `VAST_AI_GUIDE.md`
- **用途**: vast.ai云GPU平台使用指南
- **内容**: 
  - vast.ai平台介绍和注册
  - 实例创建和配置
  - 环境部署和训练流程
  - 成本优化建议
- **适用场景**: 需要使用云GPU进行训练时参考 