# 🚀 OsteoToxPred 部署指南

## 📋 项目简介

OsteoToxPred 是一个基于深度学习的骨毒性预测平台，使用多模态特征融合图神经网络（BTP-MFFGNN）进行药物分子骨毒性预测。

### 🎯 核心特性
- **高精度预测**: 准确率 0.85，AUC 0.92
- **多模型支持**: BTP-MFFGNN、SVM、RandomForest、XGBoost
- **批量预测**: 支持最多50个分子同时预测
- **可视化结果**: 分子结构图、注意力权重、热力图
- **Web界面**: 响应式设计，支持中英文

## 🛠️ 系统要求

### 最低配置
- **CPU**: 4核心
- **内存**: 8GB RAM
- **存储**: 10GB 可用空间
- **操作系统**: Ubuntu 18.04+ / CentOS 7+ / Windows 10+

### 推荐配置
- **CPU**: 8核心
- **内存**: 16GB RAM
- **存储**: 20GB SSD
- **GPU**: NVIDIA GPU (可选，用于加速)

## 📦 部署方式

### 方式1: 快速部署（推荐）

使用提供的部署脚本：

```bash
# 克隆项目
git clone <your-repo-url>
cd OsteoToxPred

# 给部署脚本执行权限
chmod +x deploy.sh

# 生产环境部署
./deploy.sh production

# 启动服务
python3 run_production.py
```

### 方式2: Docker部署（推荐服务器）

```bash
# 构建并启动
./deploy.sh docker

# 或手动执行
docker-compose up -d
```

### 方式3: 手动部署

#### 1. 安装依赖
```bash
# 安装Python依赖
pip3 install -r requirements.txt

# 创建必要目录
mkdir -p logs static/tmp flask_session
```

#### 2. 配置环境
```bash
export FLASK_ENV=production
export FLASK_APP=app.py
export SECRET_KEY="your-super-secret-key"
```

#### 3. 启动服务
```bash
# 使用Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app

# 或使用生产脚本
python3 run_production.py
```

## 🔧 配置说明

### 环境变量
```bash
FLASK_ENV=production          # 运行环境
FLASK_APP=app.py             # 应用入口
SECRET_KEY=your-secret-key   # 会话密钥
```

### 模型配置
主要配置文件：`config_flask.yaml`
- 模型参数：层数、维度、dropout等
- 数据集配置：指纹类型、位数等
- 设备配置：CPU/GPU选择

### 文件权限
确保以下目录有写权限：
- `logs/` - 日志文件
- `static/tmp/` - 临时图片
- `flask_session/` - 会话存储

## 🌐 Nginx配置

### 1. 安装Nginx
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install nginx

# CentOS/RHEL
sudo yum install nginx
```

### 2. 配置反向代理
```bash
# 复制配置文件
sudo cp nginx.conf /etc/nginx/sites-available/osteotoxpred
sudo ln -s /etc/nginx/sites-available/osteotoxpred /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启Nginx
sudo systemctl restart nginx
```

### 3. SSL配置（可选）
```bash
# 使用Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 📊 监控和维护

### 健康检查
```bash
# 检查应用状态
curl http://localhost:5000/health

# 使用部署脚本
./deploy.sh health
```

### 日志查看
```bash
# 应用日志
tail -f logs/error.log

# 使用部署脚本
./deploy.sh logs

# Docker日志
docker-compose logs -f
```

### 性能监控
```bash
# 查看进程
ps aux | grep gunicorn

# 查看端口占用
netstat -tlnp | grep 5000

# 查看资源使用
htop
```

## 🔒 安全配置

### 1. 防火墙设置
```bash
# 开放必要端口
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

### 2. 修改默认密钥
```bash
# 生成随机密钥
python3 -c "import secrets; print(secrets.token_hex(32))"

# 设置环境变量
export SECRET_KEY="your-generated-secret-key"
```

### 3. 限制文件上传
- 最大文件大小：16MB
- 支持格式：SMILES文本
- 自动清理临时文件

## 🚨 故障排除

### 常见问题

#### 1. 模型加载失败
```bash
# 检查模型文件
ls -la ckpt/checkpoints/model.pth

# 检查权限
chmod 644 ckpt/checkpoints/model.pth
```

#### 2. 端口被占用
```bash
# 查看端口占用
sudo lsof -i :5000

# 杀死进程
sudo kill -9 <PID>
```

#### 3. 内存不足
```bash
# 减少工作进程数
gunicorn -w 2 -b 0.0.0.0:5000 app:app

# 或修改run_production.py中的workers数量
```

#### 4. Redis连接失败
应用会自动回退到文件系统存储，无需额外配置。

### 日志分析
```bash
# 查看错误日志
grep "ERROR" logs/error.log

# 查看访问日志
tail -f logs/access.log
```

## 📈 性能优化

### 1. 服务器优化
- 使用SSD存储
- 增加内存容量
- 启用CPU多核处理

### 2. 应用优化
- 调整Gunicorn工作进程数
- 启用预加载应用
- 配置连接池

### 3. 数据库优化
- 使用Redis缓存
- 配置连接超时
- 启用压缩

## 🔄 更新部署

### 1. 备份数据
```bash
# 备份模型文件
cp -r ckpt/ ckpt_backup/

# 备份配置文件
cp config_flask.yaml config_flask.yaml.backup
```

### 2. 更新代码
```bash
# 拉取最新代码
git pull origin main

# 重新部署
./deploy.sh production
```

### 3. 验证更新
```bash
# 健康检查
./deploy.sh health

# 功能测试
curl -X POST http://localhost:5000/predict \
  -d "algorithm=BTP-MFFGNN&smiles=CCO"
```

## 📞 技术支持

### 联系方式
- 项目地址：[GitHub Repository]
- 问题反馈：[Issues Page]
- 邮箱：[your-email@domain.com]

### 文档资源
- API文档：`/help` 页面
- 用户手册：`/introduction` 页面
- 数据集说明：`/hwvlab` 页面

---

## 🎉 部署完成

恭喜！您的OsteoToxPred平台已成功部署。现在可以：

1. 访问 `http://your-server:5000` 使用平台
2. 进行分子骨毒性预测
3. 查看预测结果和可视化
4. 下载数据集和结果

如有任何问题，请参考故障排除部分或联系技术支持。
