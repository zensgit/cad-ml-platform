# 🚀 GitHub私有仓库设置指南

本文档指导您如何将CAD ML Platform项目设置为GitHub私有仓库。

---

## 📋 前置准备

1. **GitHub账号**: 确保您有GitHub账号
2. **Git工具**: 本地已安装Git (`git --version`)
3. **GitHub CLI** (可选): 安装GitHub CLI可简化操作 (`brew install gh` 或 `winget install gh`)

---

## 🔧 方法一：使用GitHub CLI（推荐）

### 步骤1：安装并认证GitHub CLI

```bash
# macOS
brew install gh

# Windows
winget install GitHub.cli

# Linux
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# 认证
gh auth login
```

### 步骤2：创建私有仓库

```bash
cd /Users/huazhou/Insync/hua.chau@outlook.com/OneDrive/应用/GitHub/cad-ml-platform

# 初始化Git仓库
git init

# 添加所有文件
git add .

# 初始提交
git commit -m "Initial commit: CAD ML Platform v1.0.0"

# 使用GitHub CLI创建私有仓库
gh repo create cad-ml-platform --private --source=. --remote=origin --push
```

---

## 🖱️ 方法二：通过GitHub网页界面

### 步骤1：在GitHub创建私有仓库

1. 访问 https://github.com/new
2. 填写信息：
   - Repository name: `cad-ml-platform`
   - Description: `智能CAD分析微服务平台 - Intelligent CAD Analysis Microservice Platform`
   - 选择 **Private** (私有)
   - 不要勾选 "Initialize this repository with a README"
3. 点击 "Create repository"

### 步骤2：本地初始化并推送

```bash
cd /Users/huazhou/Insync/hua.chau@outlook.com/OneDrive/应用/GitHub/cad-ml-platform

# 初始化Git仓库
git init

# 添加所有文件
git add .

# 初始提交
git commit -m "Initial commit: CAD ML Platform v1.0.0"

# 添加远程仓库（替换YOUR_USERNAME为您的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/cad-ml-platform.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

---

## 🛡️ 方法三：使用个人访问令牌（PAT）

如果您启用了两因素认证，需要使用个人访问令牌：

### 步骤1：创建个人访问令牌

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 设置：
   - Note: `CAD ML Platform Access`
   - Expiration: 选择合适的过期时间
   - Scopes: 勾选 `repo` (完整权限)
4. 点击 "Generate token"
5. **复制令牌**（只显示一次）

### 步骤2：使用令牌推送

```bash
# 使用令牌作为密码
git remote add origin https://github.com/YOUR_USERNAME/cad-ml-platform.git
git push -u origin main
# 用户名：YOUR_USERNAME
# 密码：YOUR_PERSONAL_ACCESS_TOKEN
```

---

## 📝 推荐的Git工作流程

### 分支策略

```bash
# 创建开发分支
git checkout -b develop

# 创建功能分支
git checkout -b feature/new-feature

# 创建修复分支
git checkout -b hotfix/bug-fix
```

### 提交规范

```bash
# 功能
git commit -m "feat: 添加批量分析API"

# 修复
git commit -m "fix: 修复缓存失效问题"

# 文档
git commit -m "docs: 更新API文档"

# 性能
git commit -m "perf: 优化特征提取性能"

# 重构
git commit -m "refactor: 重构适配器模式"
```

---

## 🔐 安全建议

### 1. 设置.gitignore

确保敏感信息不被提交：

```bash
# 检查.gitignore是否正确
git status --ignored

# 如果已经提交了敏感文件，从历史中删除
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config/secrets.yaml" \
  --prune-empty --tag-name-filter cat -- --all
```

### 2. 使用GitHub Secrets

对于CI/CD，使用GitHub Secrets存储敏感信息：

1. 访问仓库设置：Settings → Secrets and variables → Actions
2. 添加密钥：
   - `CADML_API_KEY`
   - `DOCKER_REGISTRY_PASSWORD`
   - `REDIS_PASSWORD`

### 3. 分支保护

设置main分支保护规则：

1. Settings → Branches
2. Add rule:
   - Branch name pattern: `main`
   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Include administrators

---

## 🤝 协作设置

### 添加协作者

1. Settings → Manage access
2. Invite a collaborator
3. 设置权限级别

### 团队开发

```bash
# 克隆私有仓库
git clone https://github.com/YOUR_USERNAME/cad-ml-platform.git

# 配置用户信息
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 创建功能分支
git checkout -b feature/your-feature

# 提交并创建Pull Request
git push origin feature/your-feature
```

---

## 📦 GitHub Actions CI/CD

创建 `.github/workflows/ci.yml`：

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/

  docker:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Build and push Docker image
      env:
        DOCKER_REGISTRY: ${{ secrets.DOCKER_REGISTRY }}
      run: |
        docker build -t cad-ml-platform:latest .
        docker push $DOCKER_REGISTRY/cad-ml-platform:latest
```

---

## 🏷️ 版本发布

### 创建Release

```bash
# 创建标签
git tag -a v1.0.0 -m "Release version 1.0.0"

# 推送标签
git push origin v1.0.0

# 使用GitHub CLI创建Release
gh release create v1.0.0 \
  --title "CAD ML Platform v1.0.0" \
  --notes "Initial release with core features" \
  --prerelease
```

---

## 📊 项目看板

设置项目看板追踪进度：

1. Projects → New project
2. 选择模板：Basic Kanban
3. 创建列：
   - To Do
   - In Progress
   - Review
   - Done

---

## 🔧 常见问题

### Q1: Permission denied

```bash
# 检查SSH密钥
ssh -T git@github.com

# 如果失败，生成新密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 添加到GitHub
cat ~/.ssh/id_ed25519.pub
# 复制内容到 GitHub Settings → SSH and GPG keys
```

### Q2: 大文件处理

```bash
# 使用Git LFS处理大文件
git lfs track "*.pkl"
git lfs track "*.h5"
git add .gitattributes
```

### Q3: 修改远程仓库URL

```bash
# 查看当前远程仓库
git remote -v

# 修改URL
git remote set-url origin https://github.com/NEW_USERNAME/cad-ml-platform.git
```

---

## 📚 相关资源

- [GitHub文档](https://docs.github.com)
- [Git教程](https://git-scm.com/book)
- [GitHub CLI文档](https://cli.github.com/manual/)

---

## ✅ 检查清单

- [ ] Git已安装并配置
- [ ] GitHub账号已创建
- [ ] 私有仓库已创建
- [ ] 本地代码已推送
- [ ] .gitignore正确配置
- [ ] 敏感信息已排除
- [ ] 协作者已添加（如需要）
- [ ] 分支保护已设置
- [ ] CI/CD已配置（可选）

---

**完成以上步骤后，您的CAD ML Platform项目就成功设置为GitHub私有仓库了！** 🎉
