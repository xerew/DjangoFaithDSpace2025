# Ollama SSH tunnel

This Compose sidecar keeps a private SSH tunnel open from the application
network to Ollama on the NTUA VM. Port 11434 is exposed only to other Compose
services; it is not published on the host.

## One-time setup on the deployment host

Run these commands on the machine that hosts the platform (for example, the
Trust VM). This ensures the persistent SSH key is stored in that Docker host's
named volume rather than on a developer workstation.

Build the sidecar and install its dedicated public key on the NTUA VM:

```powershell
docker compose build ollama-tunnel
docker compose run --rm ollama-tunnel setup
```

The second command prompts for the NTUA SSH password. The password is used by
SSH only and is not stored. The generated private key remains in the Docker
named volume `ollama_ssh`.

Then start or recreate the tunnel and Celery worker:

```powershell
docker compose up -d ollama-tunnel celery
```

## Checks

Check tunnel health:

```powershell
docker compose ps ollama-tunnel
docker compose exec celery python -c "import requests; print(requests.get('http://ollama-tunnel:11434/api/tags', timeout=10).json())"
```

The sidecar includes an internal nginx proxy that rewrites the Docker service
hostname to `Host: localhost:11434`. Current Ollama versions reject requests
whose `Host` header contains the Compose service name, even though the SSH
tunnel itself is healthy.

The sidecar uses `autossh` plus SSH keepalives, so it reconnects automatically
after transient network or VM interruptions.
