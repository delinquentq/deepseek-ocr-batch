# 🚀 通过GitHub部署到服务器 - 完整指南

## 📋 部署流程概览

```
本地开发 → GitHub仓库 → 服务器克隆 → 环境配置 → 运行系统
```

## 🔄 步骤1: 本地Git初始化和推送

### 1.1 初始化本地Git仓库

```bash
# 在项目目录下初始化Git
cd "H:\居丽叶玩具项目\DeepSeek-OCR-main\DeepSeek-OCR-master\DeepSeek-OCR-vllm"

# 初始化Git仓库
git init

# 创建.gitignore文件
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 日志文件
*.log
logs/

# 临时文件
temp_processing/
*.tmp
*.temp

# 输出文件
output_results/
input_pdfs/*.pdf

# 敏感信息
.env
*.key
*.pem

# 系统文件
.DS_Store
Thumbs.db

# 模型缓存
model_cache/
.cache/

# 测试报告
test_report.json
performance_report.json
EOF

# 添加所有文件
git add .

# 提交初始版本
git commit -m "feat: 初始化DeepSeek OCR批量处理系统

- 支持批量PDF处理
- 双模型对比 (Gemini 2.5 Flash + Qwen3-VL-30B)
- 严格JSON Schema验证
- RTX 3090 24G显存优化
- 完整的测试和监控系统"
```

### 1.2 推送到GitHub

```bash
# 在GitHub上创建新仓库: deepseek-ocr-batch
# 然后关联远程仓库

git remote add origin https://github.com/yourusername/deepseek-ocr-batch.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

## 🖥️ 步骤2: 服务器部署

### 2.1 创建一键部署脚本

首先在本地创建服务器部署脚本：