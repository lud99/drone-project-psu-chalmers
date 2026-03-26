FROM --platform=linux/amd64 nginx:latest

ENV BACKEND_URL="ws://localhost:8000"

RUN apt-get update && apt-get install -y gettext-base

COPY nginx.conf /etc/nginx/conf.d/default.conf

WORKDIR /usr/share/nginx/html

COPY index.html /usr/share/nginx/html/index.html
COPY images /usr/share/nginx/html/images
COPY misc /usr/share/nginx/html/misc

CMD ["sh", "-c", "nginx -g 'daemon off;'"]