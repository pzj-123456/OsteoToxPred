# Flask应用部署指南

## 🚀 部署方式

### 方式1: 直接部署（推荐开发环境）

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **启动应用**
```bash
python app.py
```

### 方式2: 使用Gunicorn（推荐生产环境）

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **启动应用**
```bash
python run_production.py
```

### 方式3: Docker部署（推荐服务器环境）

1. **构建镜像**
```bash
docker build -t flask-toxicity-app .
```

2. **运行容器**
```bash
docker run -d -p 5000:5000 --name toxicity-app flask-toxicity-app
```

## 🔧 环境配置

### 开发环境
```bash
export FLASK_ENV=development
export FLASK_APP=app.py
```

### 生产环境
```bash
export FLASK_ENV=production
export FLASK_APP=app.py
export SECRET_KEY=your-super-secret-key
```

## 📁 目录结构
```
FlaskProject/
├── app.py                 # 主应用文件
├── config_production.py   # 生产环境配置
├── requirements.txt       # 依赖包列表
├── run_production.py     # 生产环境启动脚本
├── Dockerfile            # Docker配置文件
├── models/               # 模型相关文件
├── static/               # 静态文件
├── templates/            # 模板文件
├── data/                 # 数据文件
└── ckpt/                # 模型检查点
```

## ⚠️ 注意事项

1. **模型文件**: 确保`ckpt/912maybe/model.pth`文件存在
2. **数据文件**: 确保`data/test.csv`文件存在
3. **权限**: 确保应用有写入`static/tmp`和`flask_session`目录的权限
4. **端口**: 默认使用5000端口，可通过环境变量修改

## 🔍 故障排除

### 常见问题

1. **Tkinter错误**: 已通过设置matplotlib后端为'Agg'解决
2. **Redis连接失败**: 自动回退到文件系统存储
3. **模型加载失败**: 检查模型文件路径和权限

### 日志查看
```bash
# 查看应用日志
tail -f logs/app.log

# 查看Docker容器日志
docker logs toxicity-app
```

## 📊 性能优化

1. **工作进程**: 默认4个工作进程，可根据服务器配置调整
2. **超时设置**: 请求超时120秒，适合长时间运行的模型预测
3. **连接限制**: 最大1000个请求，防止内存溢出

## 🔒 安全建议

1. **修改SECRET_KEY**: 生产环境必须修改默认密钥
2. **限制文件上传**: 已设置16MB文件大小限制
3. **Session管理**: 使用文件系统存储，避免Redis依赖
4. **错误信息**: 生产环境不显示详细错误信息
