
### 2. Nginx配置
FROM nginx:alpine

# 复制Nginx配置文件
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 创建SSL证书目录（可选）
# RUN mkdir -p /etc/nginx/ssl

# 暴露端口
EXPOSE 80
# EXPOSE 443

CMD ["nginx", "-g", "daemon off;"]