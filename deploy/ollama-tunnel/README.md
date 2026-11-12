# Ollama SSH tunnel

This Compose sidecar keeps a private SSH tunnel open from the application
network to Ollama on the NTUA VM. Port 11434 is exposed only to other Compose
services; it is not published on the host. The application always reaches
Ollama at `http://ollama-tunnel:11434`, whichever address the tunnel uses.

## Addresses

The NTUA machine has been reachable at different addresses over time. The
tunnel knows a list of them, `OLLAMA_SSH_HOSTS` in the deployment `.env`
(comma-separated, tried in that order). The default is:

```
OLLAMA_SSH_HOSTS=147.102.17.66,147.102.23.222
```

The older single-address setting `OLLAMA_SSH_HOST` still works: if it is set,
that address is added to the end of the list.

Admins choose how the tunnel uses the list in Django admin, under
**AI server connection** (link at the top right of every admin page,
`/admin/ai-server/`):

- **Automatic** (the default): tries the addresses in order and stays on the
  first one that connects. If the connection drops, it tries the address that
  last worked first, then the others, and keeps trying every 15 seconds if
  none answers.
- **Always use** one address: only that address is tried.

A change takes effect within about 10 seconds. The page shows which address
is connected, since when, and the last SSH error. Admins can only choose
addresses from the list; adding one needs the `.env` change and setup below.

Django and the tunnel share the `ollama_control` volume for this: Django
writes the choice to `target`, the tunnel writes what it is doing to
`status` every few seconds.

## One-time setup on the deployment host

Run these commands on the machine that hosts the platform (for example, the
Trust VM). This ensures the persistent SSH key is stored in that Docker host's
named volume rather than on a developer workstation.

Build the sidecar and install its dedicated public key on the NTUA VM:

```powershell
docker compose build ollama-tunnel
docker compose run --rm ollama-tunnel setup
```

`setup` goes through every address in the list. For each one it may prompt
for the NTUA SSH password; the password is used by SSH only and is not
stored. An address that can't be reached is skipped with a message; run
setup again when it is reachable (or `setup <address>` for just that one).
The generated private key remains in the Docker named volume `ollama_ssh`.

Then start or recreate the tunnel, web and Celery containers:

```powershell
docker compose up -d ollama-tunnel web celery
```

## Adding or changing an address

1. Edit `OLLAMA_SSH_HOSTS` in the deployment `.env`.
2. Register the new address's SSH host key (and the tunnel key, if needed):
   `docker compose run --rm ollama-tunnel setup <address>`
3. Recreate the tunnel: `docker compose up -d --no-deps ollama-tunnel`

Without step 2 the admin page shows "Host key verification failed." for that
address, and Automatic mode moves on to the next one.

Setup reuses the existing key in `ollama_ssh`; do not delete that volume.

## Checks

Check tunnel health:

```powershell
docker compose ps ollama-tunnel
docker compose logs --tail 20 ollama-tunnel
docker compose exec celery python -c "import requests; print(requests.get('http://ollama-tunnel:11434/api/tags', timeout=10).json())"
```

The sidecar includes an internal nginx proxy that rewrites the Docker service
hostname to `Host: localhost:11434`. Current Ollama versions reject requests
whose `Host` header contains the Compose service name, even though the SSH
tunnel itself is healthy.

The tunnel uses SSH keepalives (every 15 seconds, 3 misses) to notice a dead
connection, then reconnects as described under Addresses.
