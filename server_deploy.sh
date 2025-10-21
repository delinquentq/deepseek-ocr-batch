#!/bin/bash

# DeepSeek OCR 批量处理系统 - 服务器一键部署脚本
# 使用方法: curl -fsSL https://raw.githubusercontent.com/yourusername/deepseek-ocr-batch/main/server_deploy.sh | bash

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置变量
REPO_URL="https://github.com/yourusername/deepseek-ocr-batch.git"
PROJECT_DIR="$HOME/deepseek-ocr-batch"
CONDA_ENV_NAME="deepseek-ocr"
PYTHON_VERSION="3.10"

echo -e "${BLUE}🚀 DeepSeek OCR 批量处理系统 - 服务器部署${NC}"
echo -e "${BLUE}===============================================${NC}"

# 函数：检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 未安装${NC}"
        return 1
    else
        echo -e "${GREEN}✅ $1 已安装${NC}"
        return 0
    fi
}

# 函数：检查GPU
check_gpu() {
    echo -e "${BLUE}🔍 检查GPU环境...${NC}"

    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ NVIDIA驱动已安装${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

        # 检查显存
        gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        if [ $gpu_memory -ge 20000 ]; then
            echo -e "${GREEN}✅ 显存充足: ${gpu_memory}MB${NC}"
        else
            echo -e "${YELLOW}⚠️  显存较少: ${gpu_memory}MB (推荐 ≥20GB)${NC}"
        fi
    else
        echo -e "${RED}❌ 未检测到NVIDIA GPU${NC}"
        exit 1
    fi
}

# 函数：安装系统依赖
install_system_dependencies() {
    echo -e "${BLUE}📦 安装系统依赖...${NC}"

    # 检测操作系统
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
    fi

    case $OS in
        *Ubuntu*|*Debian*)
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                git \
                curl \
                wget \
                vim \
                htop \
                screen \
                tmux \
                python3-dev \
                python3-pip \
                libffi-dev \
                libssl-dev \
                libjpeg-dev \
                libpng-dev \
                libfreetype6-dev \
                pkg-config
            ;;
        *CentOS*|*RedHat*|*Fedora*)
            sudo yum update -y
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                cmake \
                git \
                curl \
                wget \
                vim \
                htop \
                screen \
                tmux \
                python3-devel \
                python3-pip \
                libffi-devel \
                openssl-devel \
                libjpeg-devel \
                libpng-devel \
                freetype-devel \
                pkgconfig
            ;;
        *)
            echo -e "${YELLOW}⚠️  未识别的操作系统: $OS${NC}"
            echo -e "${YELLOW}请手动安装必要的开发工具${NC}"
            ;;
    esac

    echo -e "${GREEN}✅ 系统依赖安装完成${NC}"
}

# 函数：安装Conda
install_conda() {
    echo -e "${BLUE}🐍 安装Miniconda...${NC}"

    if ! check_command conda; then
        # 下载并安装Miniconda
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
        bash ~/miniconda.sh -b -p $HOME/miniconda
        rm ~/miniconda.sh

        # 添加到PATH
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/miniconda/bin:$PATH"

        # 初始化conda
        conda init bash

        echo -e "${GREEN}✅ Miniconda安装完成${NC}"
    else
        echo -e "${GREEN}✅ Conda已存在，跳过安装${NC}"
    fi
}

# 函数：创建Conda环境
create_conda_env() {
    echo -e "${BLUE}🔧 创建Conda环境: $CONDA_ENV_NAME${NC}"

    # 检查环境是否已存在
    if conda env list | grep -q $CONDA_ENV_NAME; then
        echo -e "${YELLOW}⚠️  环境 $CONDA_ENV_NAME 已存在，是否重新创建？ (y/N)${NC}"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            conda env remove -n $CONDA_ENV_NAME -y
        else
            echo -e "${BLUE}使用现有环境${NC}"
            return 0
        fi
    fi

    # 创建新环境
    conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
    echo -e "${GREEN}✅ Conda环境创建完成${NC}"
}

# 函数：克隆项目
clone_project() {
    echo -e "${BLUE}📥 克隆项目从GitHub...${NC}"

    if [ -d "$PROJECT_DIR" ]; then
        echo -e "${YELLOW}⚠️  项目目录已存在，是否更新？ (y/N)${NC}"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            cd $PROJECT_DIR
            git pull origin main
        else
            echo -e "${BLUE}使用现有项目目录${NC}"
        fi
    else
        git clone $REPO_URL $PROJECT_DIR
    fi

    cd $PROJECT_DIR
    echo -e "${GREEN}✅ 项目克隆完成${NC}"
}

# 函数：安装Python依赖
install_python_dependencies() {
    echo -e "${BLUE}📦 安装Python依赖...${NC}"

    # 激活conda环境
    source $HOME/miniconda/bin/activate $CONDA_ENV_NAME

    # 升级pip
    pip install --upgrade pip

    # 安装PyTorch (CUDA版本)
    echo -e "${CYAN}安装PyTorch CUDA版本...${NC}"
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

    # 安装其他依赖
    echo -e "${CYAN}安装其他依赖包...${NC}"
    pip install -r requirements_batch.txt

    # 单独安装flash-attention（可能需要编译）
    echo -e "${CYAN}安装Flash Attention...${NC}"
    pip install flash-attn==2.7.3 --no-build-isolation

    echo -e "${GREEN}✅ Python依赖安装完成${NC}"
}

# 函数：配置环境
configure_environment() {
    echo -e "${BLUE}⚙️  配置运行环境...${NC}"

    # 创建必要目录
    mkdir -p input_pdfs output_results temp_processing logs

    # 设置权限
    chmod +x *.sh

    # 创建环境变量文件
    cat > .env << 'EOF'
# DeepSeek OCR 批量处理系统环境变量

# OpenRouter API配置
OPENROUTER_API_KEY=your_openrouter_api_key_here

# DeepSeek OCR模型路径
DEEPSEEK_OCR_MODEL_PATH=deepseek-ai/DeepSeek-OCR

# CUDA配置
CUDA_VISIBLE_DEVICES=0
VLLM_USE_V1=0

# 可选：Triton配置 (CUDA 11.8)
# TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas

# 系统配置
PYTHONPATH=/home/user/deepseek-ocr-batch
OMP_NUM_THREADS=8
EOF

    echo -e "${GREEN}✅ 环境配置完成${NC}"
}

# 函数：运行测试
run_tests() {
    echo -e "${BLUE}🧪 运行系统测试...${NC}"

    # 激活conda环境
    source $HOME/miniconda/bin/activate $CONDA_ENV_NAME

    # 加载环境变量
    if [ -f .env ]; then
        source .env
    fi

    # 运行基础测试
    python test_batch_system.py

    echo -e "${GREEN}✅ 系统测试完成${NC}"
}

# 函数：创建服务脚本
create_service_scripts() {
    echo -e "${BLUE}📝 创建服务脚本...${NC}"

    # 创建启动脚本
    cat > start_service.sh << EOF
#!/bin/bash

# 激活conda环境
source \$HOME/miniconda/bin/activate $CONDA_ENV_NAME

# 加载环境变量
if [ -f .env ]; then
    source .env
fi

# 检查API密钥
if [ -z "\$OPENROUTER_API_KEY" ]; then
    echo "❌ 请在.env文件中设置OPENROUTER_API_KEY"
    exit 1
fi

# 启动批量处理器
cd $PROJECT_DIR
python run_batch_processor.py "\$@"
EOF

    # 创建监控脚本
    cat > monitor_service.sh << 'EOF'
#!/bin/bash

# 监控GPU使用情况
watch -n 5 'echo "=== GPU状态 ==="; nvidia-smi; echo "=== 处理日志 ==="; tail -10 logs/batch_processor.log'
EOF

    # 创建停止脚本
    cat > stop_service.sh << 'EOF'
#!/bin/bash

# 查找并停止Python进程
pkill -f "python.*run_batch_processor.py"
echo "批量处理服务已停止"
EOF

    chmod +x *.sh

    echo -e "${GREEN}✅ 服务脚本创建完成${NC}"
}

# 函数：创建systemd服务（可选）
create_systemd_service() {
    echo -e "${BLUE}🔧 是否创建systemd服务？ (y/N)${NC}"
    read -r response

    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        sudo tee /etc/systemd/system/deepseek-ocr.service > /dev/null << EOF
[Unit]
Description=DeepSeek OCR Batch Processing Service
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$HOME/miniconda/bin:$PATH
ExecStart=$HOME/miniconda/envs/$CONDA_ENV_NAME/bin/python run_batch_processor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

        sudo systemctl daemon-reload
        sudo systemctl enable deepseek-ocr

        echo -e "${GREEN}✅ systemd服务创建完成${NC}"
        echo -e "${CYAN}使用以下命令管理服务:${NC}"
        echo -e "  启动: sudo systemctl start deepseek-ocr"
        echo -e "  停止: sudo systemctl stop deepseek-ocr"
        echo -e "  状态: sudo systemctl status deepseek-ocr"
        echo -e "  日志: sudo journalctl -u deepseek-ocr -f"
    fi
}

# 主部署流程
main() {
    echo -e "${BLUE}开始部署流程...${NC}\n"

    # 1. 检查环境
    echo -e "${CYAN}=== 步骤1: 环境检查 ===${NC}"
    check_gpu
    echo

    # 2. 安装系统依赖
    echo -e "${CYAN}=== 步骤2: 安装系统依赖 ===${NC}"
    install_system_dependencies
    echo

    # 3. 安装Conda
    echo -e "${CYAN}=== 步骤3: 安装Conda ===${NC}"
    install_conda
    echo

    # 4. 创建Conda环境
    echo -e "${CYAN}=== 步骤4: 创建Python环境 ===${NC}"
    create_conda_env
    echo

    # 5. 克隆项目
    echo -e "${CYAN}=== 步骤5: 克隆项目 ===${NC}"
    clone_project
    echo

    # 6. 安装Python依赖
    echo -e "${CYAN}=== 步骤6: 安装Python依赖 ===${NC}"
    install_python_dependencies
    echo

    # 7. 配置环境
    echo -e "${CYAN}=== 步骤7: 配置环境 ===${NC}"
    configure_environment
    echo

    # 8. 创建服务脚本
    echo -e "${CYAN}=== 步骤8: 创建服务脚本 ===${NC}"
    create_service_scripts
    echo

    # 9. 运行测试
    echo -e "${CYAN}=== 步骤9: 运行测试 ===${NC}"
    if [ "$1" != "--skip-test" ]; then
        run_tests
    else
        echo -e "${YELLOW}跳过测试${NC}"
    fi
    echo

    # 10. 创建systemd服务（可选）
    echo -e "${CYAN}=== 步骤10: 系统服务配置 ===${NC}"
    create_systemd_service
    echo

    # 部署完成
    echo -e "${GREEN}🎉 部署完成！${NC}"
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${CYAN}下一步操作:${NC}"
    echo -e "1. 编辑环境配置: ${YELLOW}vim $PROJECT_DIR/.env${NC}"
    echo -e "2. 设置API密钥: ${YELLOW}OPENROUTER_API_KEY=your_key${NC}"
    echo -e "3. 放入PDF文件: ${YELLOW}$PROJECT_DIR/input_pdfs/${NC}"
    echo -e "4. 启动处理: ${YELLOW}cd $PROJECT_DIR && ./start_service.sh${NC}"
    echo -e "5. 监控状态: ${YELLOW}./monitor_service.sh${NC}"
    echo -e "${BLUE}===============================================${NC}"
}

# 检查参数
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "DeepSeek OCR 批量处理系统 - 服务器部署脚本"
    echo ""
    echo "使用方法:"
    echo "  $0                 # 完整部署"
    echo "  $0 --skip-test     # 跳过测试"
    echo "  $0 --help          # 显示帮助"
    echo ""
    exit 0
fi

# 执行主流程
main "$@"