#!/bin/bash

# CAD ML Platform - GitHub仓库初始化脚本
# 使用方法: bash scripts/init_github.sh

set -e

echo "🚀 CAD ML Platform - GitHub仓库初始化"
echo "======================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Git是否安装
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git未安装，请先安装Git${NC}"
    echo "   macOS: brew install git"
    echo "   Ubuntu: sudo apt-get install git"
    echo "   Windows: https://git-scm.com/downloads"
    exit 1
fi

echo -e "${GREEN}✅ Git已安装${NC}"

# 检查是否在项目根目录
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo -e "${RED}❌ 请在项目根目录运行此脚本${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 当前在项目根目录${NC}"

# 检查是否已经是Git仓库
if [ -d .git ]; then
    echo -e "${YELLOW}⚠️  目录已经是Git仓库${NC}"
    read -p "是否继续？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
else
    # 初始化Git仓库
    echo "📦 初始化Git仓库..."
    git init
    echo -e "${GREEN}✅ Git仓库初始化完成${NC}"
fi

# 配置Git用户信息（如果未配置）
if [ -z "$(git config user.name)" ]; then
    read -p "请输入Git用户名: " git_name
    git config user.name "$git_name"
fi

if [ -z "$(git config user.email)" ]; then
    read -p "请输入Git邮箱: " git_email
    git config user.email "$git_email"
fi

echo -e "${GREEN}✅ Git用户信息已配置${NC}"

# 添加文件到暂存区
echo "📂 添加文件到Git..."
git add .
echo -e "${GREEN}✅ 文件已添加${NC}"

# 创建初始提交
echo "💾 创建初始提交..."
git commit -m "Initial commit: CAD ML Platform v1.0.0

- 完整的微服务架构
- CAD文件分析API
- 机器学习集成
- Docker容器化支持
- Python客户端SDK
- 完整文档" || true

echo -e "${GREEN}✅ 初始提交完成${NC}"

# 选择创建仓库的方式
echo ""
echo "请选择创建GitHub仓库的方式:"
echo "1) 使用GitHub CLI (推荐)"
echo "2) 手动在GitHub网站创建"
echo "3) 跳过GitHub设置"

read -p "请选择 (1-3): " choice

case $choice in
    1)
        # 检查GitHub CLI
        if ! command -v gh &> /dev/null; then
            echo -e "${YELLOW}⚠️  GitHub CLI未安装${NC}"
            echo "安装方法:"
            echo "   macOS: brew install gh"
            echo "   Windows: winget install GitHub.cli"
            echo "   Linux: 查看 https://cli.github.com"
            echo ""
            echo "安装后请运行: gh auth login"
            exit 0
        fi

        # 检查是否已登录
        if ! gh auth status &> /dev/null; then
            echo "需要先登录GitHub:"
            gh auth login
        fi

        # 获取GitHub用户名
        github_user=$(gh api user --jq .login)
        echo -e "${GREEN}✅ 已登录为: $github_user${NC}"

        # 创建私有仓库
        echo "🌐 创建GitHub私有仓库..."
        gh repo create cad-ml-platform \
            --private \
            --source=. \
            --description="智能CAD分析微服务平台 - Intelligent CAD Analysis Microservice Platform" \
            --remote=origin \
            --push

        echo -e "${GREEN}✅ GitHub仓库创建成功！${NC}"
        echo ""
        echo "仓库地址: https://github.com/$github_user/cad-ml-platform"
        ;;

    2)
        # 获取GitHub用户名
        read -p "请输入您的GitHub用户名: " github_user

        echo ""
        echo "请按照以下步骤操作:"
        echo "1. 访问 https://github.com/new"
        echo "2. 仓库名称: cad-ml-platform"
        echo "3. 选择 Private (私有)"
        echo "4. 不要勾选 'Initialize this repository with a README'"
        echo "5. 点击 'Create repository'"
        echo ""
        echo "创建完成后，运行以下命令:"
        echo ""
        echo -e "${YELLOW}git remote add origin https://github.com/$github_user/cad-ml-platform.git${NC}"
        echo -e "${YELLOW}git branch -M main${NC}"
        echo -e "${YELLOW}git push -u origin main${NC}"
        ;;

    3)
        echo "跳过GitHub设置"
        ;;

    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo "======================================="
echo -e "${GREEN}🎉 初始化完成！${NC}"
echo ""
echo "下一步操作:"
echo "1. 查看项目文档: cat README.md"
echo "2. 安装依赖: pip install -r requirements.txt"
echo "3. 启动服务: docker-compose up -d"
echo "4. 访问API文档: http://localhost:8000/docs"
echo ""
echo "更多信息请查看 GITHUB_SETUP.md"