#!/bin/sh
set -eu

SSH_HOST="${SSH_HOST:-147.102.17.66}"
SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-ngrammatikos}"
REMOTE_OLLAMA_HOST="${REMOTE_OLLAMA_HOST:-127.0.0.1}"
REMOTE_OLLAMA_PORT="${REMOTE_OLLAMA_PORT:-11434}"
LOCAL_OLLAMA_PORT="${LOCAL_OLLAMA_PORT:-11434}"
TUNNEL_OLLAMA_PORT="${TUNNEL_OLLAMA_PORT:-11435}"
SSH_DIR=/root/.ssh
KEY_PATH="${SSH_DIR}/id_ed25519"

mkdir -p "${SSH_DIR}"
chmod 0700 "${SSH_DIR}"

if [ "${1:-}" = "setup" ]; then
    if [ ! -s "${KEY_PATH}" ]; then
        echo "Creating a dedicated SSH key for the Ollama tunnel..."
        ssh-keygen -q -t ed25519 -N "" -C "ollama-tunnel@trust-ai" -f "${KEY_PATH}"
    fi

    echo "Installing the tunnel public key on ${SSH_USER}@${SSH_HOST}."
    echo "Enter your NTUA SSH password when prompted. It will not be stored."
    ssh-copy-id \
        -i "${KEY_PATH}.pub" \
        -p "${SSH_PORT}" \
        -o StrictHostKeyChecking=accept-new \
        "${SSH_USER}@${SSH_HOST}"
    echo "SSH tunnel setup complete."
    exit 0
fi

while [ ! -s "${KEY_PATH}" ]; do
    echo "Ollama tunnel is waiting for its SSH key."
    echo "Run: docker compose run --rm ollama-tunnel setup"
    sleep 15
done

chmod 0600 "${KEY_PATH}"

for port_value in \
    "${REMOTE_OLLAMA_PORT}" \
    "${LOCAL_OLLAMA_PORT}" \
    "${TUNNEL_OLLAMA_PORT}"
do
    case "${port_value}" in
        ''|*[!0-9]*)
            echo "Ollama tunnel ports must be numeric." >&2
            exit 1
            ;;
    esac
done

sed \
    -e "s/__LOCAL_OLLAMA_PORT__/${LOCAL_OLLAMA_PORT}/g" \
    -e "s/__TUNNEL_OLLAMA_PORT__/${TUNNEL_OLLAMA_PORT}/g" \
    -e "s/__REMOTE_OLLAMA_PORT__/${REMOTE_OLLAMA_PORT}/g" \
    /etc/nginx/nginx.conf.template \
    > /etc/nginx/nginx.conf

nginx -t
nginx

export AUTOSSH_GATETIME=0
export AUTOSSH_POLL=30

echo "Opening Ollama tunnel through ${SSH_USER}@${SSH_HOST}:${SSH_PORT}..."
echo "Rewriting Docker Host headers through nginx on port ${LOCAL_OLLAMA_PORT}."
autossh \
    -M 0 \
    -N \
    -T \
    -p "${SSH_PORT}" \
    -i "${KEY_PATH}" \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=yes \
    -L "127.0.0.1:${TUNNEL_OLLAMA_PORT}:${REMOTE_OLLAMA_HOST}:${REMOTE_OLLAMA_PORT}" \
    "${SSH_USER}@${SSH_HOST}" &
AUTOSSH_PID=$!

shutdown() {
    kill "${AUTOSSH_PID}" 2>/dev/null || true
    nginx -s quit 2>/dev/null || true
}

trap shutdown INT TERM

status=0
wait "${AUTOSSH_PID}" || status=$?
nginx -s quit 2>/dev/null || true
exit "${status}"
