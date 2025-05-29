FROM nginx:latest

ENV BACKEND_URL="ws://localhost:8000"

# Inject backend URL into frontend HTML
COPY webapp.html /webapp.html
RUN envsubst < /webapp.html > /usr/share/nginx/html/index.html
COPY misc/ /usr/share/nginx/html/
COPY images/ /usr/share/nginx/html/images

# Add nginx config to route API calls to backend
COPY nginx.conf /etc/nginx/conf.d/default.conf