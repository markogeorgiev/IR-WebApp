FROM nginx:1.25-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY frontend/*.html /usr/share/nginx/html/
COPY static /usr/share/nginx/html/static
