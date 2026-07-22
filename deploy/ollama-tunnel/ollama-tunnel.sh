#!/bin/sh
set -eu

SSH_HOST="${SSH_HOST:-147.102.17.66}"
SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-ngrammatikos}"
REMOTE_OLLAMA_HOST="${REMOTE_OLLAMA_HOST:-127.0.0.1}"
REMOTE_OLLAMA_PORT="${REMOTE_OLLAMA_PORT:-11434}"
LOCAL_OLLAMA_PORT="${LOCAL_OLLAMA_PORT:-11434}"
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

export AUTOSSH_GATETIME=0
export AUTOSSH_POLL=30

echo "Opening Ollama tunnel through ${SSH_USER}@${SSH_HOST}:${SSH_PORT}..."
exec autossh \
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
    -L "0.0.0.0:${LOCAL_OLLAMA_PORT}:${REMOTE_OLLAMA_HOST}:${REMOTE_OLLAMA_PORT}" \
    "${SSH_USER}@${SSH_HOST}"
