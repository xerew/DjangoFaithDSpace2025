FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        autossh \
        ca-certificates \
        curl \
        nginx \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

COPY nginx.conf.template /etc/nginx/nginx.conf.template
COPY ollama-tunnel.sh /usr/local/bin/ollama-tunnel
RUN chmod 0755 /usr/local/bin/ollama-tunnel

ENTRYPOINT ["/usr/local/bin/ollama-tunnel"]
