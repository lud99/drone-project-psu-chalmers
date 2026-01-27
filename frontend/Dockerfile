FROM nginx:latest

ENV BACKEND_URL="ws://localhost:8000"

# Install envsubst if not present
RUN apt-get update && apt-get install -y gettext-base

# Copy only nginx config, rest is accessed via a volume
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Set working directory
WORKDIR /usr/share/nginx/html

WORKDIR /usr/share/nginx/html
CMD ["sh", "-c", "nginx -g 'daemon off;'"]


# TODO: do this in the future, but replace it in a file we never change, that other javascript files can use
# RUN envsubst < /webapp.html > /usr/share/nginx/html/index.html