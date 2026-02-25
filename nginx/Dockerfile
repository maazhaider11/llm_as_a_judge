FROM nginx:alpine
LABEL authors="issm"

COPY ./conf /etc/nginx/conf.d
COPY ./index.html /usr/share/nginx/html/index.html

# Copy SSL certificates if SSL is configured
#COPY ./ssl /etc/ssl
