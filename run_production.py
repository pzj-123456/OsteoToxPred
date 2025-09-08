#!/usr/bin/env python3
"""
生产环境启动脚本
使用Gunicorn作为WSGI服务器
"""

import os
import sys
import multiprocessing

# 设置环境变量
os.environ['FLASK_ENV'] = 'production'
os.environ['FLASK_APP'] = 'app.py'

# 获取CPU核心数
cpu_count = multiprocessing.cpu_count()

# Gunicorn配置
bind = "0.0.0.0:5000"
workers = min(cpu_count * 2 + 1, 8)  # 限制最大工作进程数
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True

# 日志配置
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程名称
proc_name = "osteotoxpred"

# 创建日志目录
os.makedirs("logs", exist_ok=True)

if __name__ == "__main__":
    # 如果直接运行此脚本，启动Gunicorn
    from gunicorn.app.wsgiapp import WSGIApplication
    
    class StandaloneApplication(WSGIApplication):
        def init(self, parser, opts, args):
            self.cfg.set("bind", bind)
            self.cfg.set("workers", workers)
            self.cfg.set("worker_class", worker_class)
            self.cfg.set("worker_connections", worker_connections)
            self.cfg.set("timeout", timeout)
            self.cfg.set("keepalive", keepalive)
            self.cfg.set("max_requests", max_requests)
            self.cfg.set("max_requests_jitter", max_requests_jitter)
            self.cfg.set("preload_app", preload_app)
            self.cfg.set("accesslog", accesslog)
            self.cfg.set("errorlog", errorlog)
            self.cfg.set("loglevel", loglevel)
            self.cfg.set("access_log_format", access_log_format)
            self.cfg.set("proc_name", proc_name)
    
    StandaloneApplication("app:app").run()
