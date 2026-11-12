#!/bin/sh
set -eu

# Addresses of the NTUA machine, tried in this order (commas or spaces).
# SSH_HOST is the older single-address setting; if a deployment still sets
# it, that address is added to the end of the list.
SSH_HOSTS="${SSH_HOSTS:-147.102.17.66}"
SSH_HOST="${SSH_HOST:-}"
SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-ngrammatikos}"
REMOTE_OLLAMA_HOST="${REMOTE_OLLAMA_HOST:-127.0.0.1}"
REMOTE_OLLAMA_PORT="${REMOTE_OLLAMA_PORT:-11434}"
LOCAL_OLLAMA_PORT="${LOCAL_OLLAMA_PORT:-11434}"
TUNNEL_OLLAMA_PORT="${TUNNEL_OLLAMA_PORT:-11435}"
# Shared with Django (OLLAMA_TUNNEL_CONTROL_DIR): Django writes `target`
# ("auto" or one of the addresses), this script writes `status`.
CONTROL_DIR="${CONTROL_DIR:-/control}"
CONNECT_WAIT="${CONNECT_WAIT:-20}"
RETRY_WAIT="${RETRY_WAIT:-15}"
CHECK_EVERY="${CHECK_EVERY:-5}"
SSH_DIR=/root/.ssh
KEY_PATH="${SSH_DIR}/id_ed25519"
SSH_LOG=/tmp/ollama-tunnel-ssh.log

HOSTS=""
for host in $(printf '%s %s' "${SSH_HOSTS}" "${SSH_HOST}" | tr ',' ' '); do
    case "${host}" in
        *[!A-Za-z0-9.:-]*)
            echo "Ollama tunnel: invalid address '${host}'." >&2
            exit 1
            ;;
    esac
    case " ${HOSTS} " in
        *" ${host} "*) ;;
        *) HOSTS="${HOSTS:+${HOSTS} }${host}" ;;
    esac
done
if [ -z "${HOSTS}" ]; then
    echo "Ollama tunnel: no address set in SSH_HOSTS." >&2
    exit 1
fi

mkdir -p "${SSH_DIR}"
chmod 0700 "${SSH_DIR}"

if [ "${1:-}" = "setup" ]; then
    if [ ! -s "${KEY_PATH}" ]; then
        echo "Creating a dedicated SSH key for the Ollama tunnel..."
        ssh-keygen -q -t ed25519 -N "" -C "ollama-tunnel@trust-ai" -f "${KEY_PATH}"
    fi

    # `setup` registers every address; `setup <address>` only that one.
    targets="${2:-${HOSTS}}"
    done_count=0
    for host in ${targets}; do
        echo "Installing the tunnel public key on ${SSH_USER}@${host}."
        echo "Enter your NTUA SSH password if prompted. It will not be stored."
        if ssh-copy-id \
            -i "${KEY_PATH}.pub" \
            -p "${SSH_PORT}" \
            -o ConnectTimeout=10 \
            -o StrictHostKeyChecking=accept-new \
            "${SSH_USER}@${host}"
        then
            done_count=$((done_count + 1))
        else
            echo "Could not reach ${host} now; skipped. Run setup again when it is reachable."
        fi
    done
    if [ "${done_count}" -eq 0 ]; then
        echo "SSH tunnel setup failed for every address." >&2
        exit 1
    fi
    echo "SSH tunnel setup complete."
    exit 0
fi

for port_value in \
    "${REMOTE_OLLAMA_PORT}" \
    "${LOCAL_OLLAMA_PORT}" \
    "${TUNNEL_OLLAMA_PORT}" \
    "${CONNECT_WAIT}" \
    "${RETRY_WAIT}" \
    "${CHECK_EVERY}"
do
    case "${port_value}" in
        ''|*[!0-9]*)
            echo "Ollama tunnel ports and waits must be numbers." >&2
            exit 1
            ;;
    esac
done

mkdir -p "${CONTROL_DIR}"

STATE=waiting
CURRENT_HOST=""
MODE=auto
SINCE=$(date +%s)
MESSAGE=""
LAST_GOOD=""
SSH_PID=""

write_status() {
    tmp="${CONTROL_DIR}/.status.$$"
    {
        echo "state=${STATE}"
        echo "host=${CURRENT_HOST}"
        echo "mode=${MODE}"
        echo "hosts=$(echo "${HOSTS}" | tr ' ' ',')"
        echo "since=${SINCE}"
        echo "updated=$(date +%s)"
        echo "message=${MESSAGE}"
    } > "${tmp}"
    chmod 0644 "${tmp}"
    mv -f "${tmp}" "${CONTROL_DIR}/status"
}

# The admin's choice, or "auto" when there is none or it isn't allowed.
read_target() {
    chosen=""
    if [ -r "${CONTROL_DIR}/target" ]; then
        chosen=$(head -n 1 "${CONTROL_DIR}/target" | tr -d ' \t\r')
    fi
    case " ${HOSTS} " in
        *" ${chosen} "*) echo "${chosen}" ;;
        *) echo auto ;;
    esac
}

# Addresses to try now: the fixed one, or every one with the last
# address that worked first.
candidates() {
    if [ "${MODE}" != auto ]; then
        echo "${MODE}"
        return
    fi
    list="${LAST_GOOD}"
    for host in ${HOSTS}; do
        if [ "${host}" != "${LAST_GOOD}" ]; then
            list="${list:+${list} }${host}"
        fi
    done
    echo "${list}"
}

start_ssh() {
    : > "${SSH_LOG}"
    ssh \
        -N \
        -T \
        -p "${SSH_PORT}" \
        -i "${KEY_PATH}" \
        -o BatchMode=yes \
        -o ConnectTimeout=10 \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o StrictHostKeyChecking=yes \
        -L "127.0.0.1:${TUNNEL_OLLAMA_PORT}:${REMOTE_OLLAMA_HOST}:${REMOTE_OLLAMA_PORT}" \
        "${SSH_USER}@$1" 2>>"${SSH_LOG}" &
    SSH_PID=$!
}

stop_ssh() {
    if [ -n "${SSH_PID}" ]; then
        kill "${SSH_PID}" 2>/dev/null || true
        wait "${SSH_PID}" 2>/dev/null || true
        SSH_PID=""
    fi
}

ssh_alive() {
    [ -n "${SSH_PID}" ] && kill -0 "${SSH_PID}" 2>/dev/null
}

# SSH binds the local end of the tunnel only after logging in, so a
# listening port means the connection is up (curl exit 7 = refused).
tunnel_listening() {
    code=0
    curl --silent --output /dev/null --max-time 3 \
        "http://127.0.0.1:${TUNNEL_OLLAMA_PORT}/" || code=$?
    [ "${code}" -ne 7 ]
}

last_error() {
    tail -n 1 "${SSH_LOG}" 2>/dev/null | tr -d '\r' | cut -c1-300
}

shutdown() {
    stop_ssh
    STATE=stopped
    CURRENT_HOST=""
    MESSAGE=""
    write_status || true
    nginx -s quit 2>/dev/null || true
    exit 0
}

trap shutdown INT TERM

while [ ! -s "${KEY_PATH}" ]; do
    echo "Ollama tunnel is waiting for its SSH key."
    echo "Run: docker compose run --rm ollama-tunnel setup"
    MESSAGE="The tunnel has no SSH key yet. Run: docker compose run --rm ollama-tunnel setup"
    write_status
    sleep 15
done
chmod 0600 "${KEY_PATH}"
MESSAGE=""

sed \
    -e "s/__LOCAL_OLLAMA_PORT__/${LOCAL_OLLAMA_PORT}/g" \
    -e "s/__TUNNEL_OLLAMA_PORT__/${TUNNEL_OLLAMA_PORT}/g" \
    -e "s/__REMOTE_OLLAMA_PORT__/${REMOTE_OLLAMA_PORT}/g" \
    /etc/nginx/nginx.conf.template \
    > /etc/nginx/nginx.conf

nginx -t
nginx

echo "Ollama tunnel addresses: ${HOSTS}"
echo "Rewriting Docker Host headers through nginx on port ${LOCAL_OLLAMA_PORT}."

while :; do
    MODE=$(read_target)
    connected=0
    for host in $(candidates); do
        STATE=connecting
        CURRENT_HOST="${host}"
        SINCE=$(date +%s)
        write_status
        echo "Opening Ollama tunnel through ${SSH_USER}@${host}:${SSH_PORT} (mode: ${MODE})..."
        start_ssh "${host}"

        waited=0
        while ssh_alive && ! tunnel_listening && [ "${waited}" -lt "${CONNECT_WAIT}" ]; do
            sleep 1
            waited=$((waited + 1))
        done

        if ssh_alive && tunnel_listening; then
            STATE=connected
            SINCE=$(date +%s)
            MESSAGE=""
            LAST_GOOD="${host}"
            connected=1
            write_status
            echo "Connected through ${host}."
            # Stay until the connection drops or the admin chooses again.
            while ssh_alive; do
                sleep "${CHECK_EVERY}"
                if [ "$(read_target)" != "${MODE}" ]; then
                    echo "The address choice changed; reconnecting."
                    break
                fi
                write_status
            done
            if ! ssh_alive; then
                MESSAGE=$(last_error)
                echo "Connection through ${host} dropped: ${MESSAGE}"
            fi
            stop_ssh
            break
        fi

        MESSAGE=$(last_error)
        if [ -z "${MESSAGE}" ]; then
            MESSAGE="No answer from ${host} within ${CONNECT_WAIT} seconds."
        fi
        echo "Could not connect through ${host}: ${MESSAGE}"
        stop_ssh
        if [ "$(read_target)" != "${MODE}" ]; then
            break
        fi
    done

    if [ "${connected}" -eq 0 ]; then
        STATE=waiting
        CURRENT_HOST=""
        SINCE=$(date +%s)
        write_status
        waited=0
        while [ "${waited}" -lt "${RETRY_WAIT}" ] && [ "$(read_target)" = "${MODE}" ]; do
            sleep 1
            waited=$((waited + 1))
            if [ $((waited % CHECK_EVERY)) -eq 0 ]; then
                write_status
            fi
        done
    fi
done
