# Nginx and Let's Encrypt

The default Compose configuration remains HTTP-only for local development. The
production override enables HTTPS for `trust-ai-lab.eu` and
`www.trust-ai-lab.eu`.

## First certificate

Point both DNS names at the deployment host and make ports 80 and 443 publicly
reachable. Then start the HTTP configuration and request the certificate:

```sh
docker compose up -d nginx
docker compose run --rm --entrypoint certbot certbot certonly \
  --webroot --webroot-path /var/www/certbot \
  --email YOUR_EMAIL_ADDRESS --agree-tos --no-eff-email \
  -d trust-ai-lab.eu -d www.trust-ai-lab.eu
```

Start production with both Compose files:

```sh
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Automatic renewal

The `certbot` service checks for renewable certificates every 12 hours. The
production Nginx entrypoint checks the certificate contents once a minute and
validates and reloads Nginx when Certbot installs a new certificate. This avoids
mounting the Docker socket or restarting the container.

Test the renewal path without replacing the certificate:

```sh
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  run --rm --entrypoint certbot certbot renew --dry-run
```
