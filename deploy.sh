#!/bin/bash

# OsteoToxPred 部署脚本
# 使用方法: ./deploy.sh [production|development|docker]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 未安装"
        exit 1
    fi
    
    log_success "系统依赖检查完成"
}

# 安装Python依赖
install_dependencies() {
    log_info "安装Python依赖..."
    pip3 install -r requirements.txt
    log_success "Python依赖安装完成"
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    mkdir -p logs
    mkdir -p static/tmp
    mkdir -p flask_session
    chmod 755 logs static/tmp flask_session
    log_success "目录创建完成"
}

# 检查模型文件
check_model_files() {
    log_info "检查模型文件..."
    
    if [ ! -f "ckpt/checkpoints/model.pth" ]; then
        log_error "主模型文件不存在: ckpt/checkpoints/model.pth"
        exit 1
    fi
    
    if [ ! -f "data/bonetox-new.csv" ]; then
        log_warning "数据集文件不存在: data/bonetox-new.csv"
    fi
    
    log_success "模型文件检查完成"
}

# 开发环境部署
deploy_development() {
    log_info "部署到开发环境..."
    
    check_dependencies
    install_dependencies
    create_directories
    check_model_files
    
    # 设置环境变量
    export FLASK_ENV=development
    export FLASK_APP=app.py
    
    log_success "开发环境部署完成"
    log_info "启动命令: python3 app.py"
}

# 生产环境部署
deploy_production() {
    log_info "部署到生产环境..."
    
    check_dependencies
    install_dependencies
    create_directories
    check_model_files
    
    # 设置环境变量
    export FLASK_ENV=production
    export FLASK_APP=app.py
    export SECRET_KEY="your-super-secret-key-$(date +%s)"
    
    log_success "生产环境部署完成"
    log_info "启动命令: python3 run_production.py"
    log_info "或使用: gunicorn -w 4 -b 0.0.0.0:5000 app:app"
}

# Docker部署
deploy_docker() {
    log_info "使用Docker部署..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装"
        exit 1
    fi
    
    # 检查docker-compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose 未安装"
        exit 1
    fi
    
    # 构建镜像
    log_info "构建Docker镜像..."
    docker build -t osteotoxpred:latest .
    
    # 启动服务
    log_info "启动Docker服务..."
    docker-compose up -d
    
    log_success "Docker部署完成"
    log_info "访问地址: http://localhost:5000"
    log_info "查看日志: docker-compose logs -f"
}

# 停止服务
stop_services() {
    log_info "停止服务..."
    
    if [ -f "docker-compose.yml" ]; then
        docker-compose down
        log_success "Docker服务已停止"
    fi
    
    # 停止Gunicorn进程
    pkill -f "gunicorn.*app:app" || true
    pkill -f "python.*run_production.py" || true
    
    log_success "所有服务已停止"
}

# 查看日志
view_logs() {
    log_info "查看应用日志..."
    
    if [ -f "logs/error.log" ]; then
        tail -f logs/error.log
    else
        log_warning "日志文件不存在"
    fi
}

# 健康检查
health_check() {
    log_info "执行健康检查..."
    
    if curl -f http://localhost:5000/health > /dev/null 2>&1; then
        log_success "应用运行正常"
    else
        log_error "应用运行异常"
        exit 1
    fi
}

# 主函数
main() {
    case "${1:-production}" in
        "development")
            deploy_development
            ;;
        "production")
            deploy_production
            ;;
        "docker")
            deploy_docker
            ;;
        "stop")
            stop_services
            ;;
        "logs")
            view_logs
            ;;
        "health")
            health_check
            ;;
        *)
            echo "使用方法: $0 [development|production|docker|stop|logs|health]"
            echo ""
            echo "命令说明:"
            echo "  development  - 部署到开发环境"
            echo "  production   - 部署到生产环境"
            echo "  docker       - 使用Docker部署"
            echo "  stop         - 停止所有服务"
            echo "  logs         - 查看应用日志"
            echo "  health       - 健康检查"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
