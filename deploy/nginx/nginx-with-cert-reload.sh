#!/bin/sh
set -eu

certificate=/etc/letsencrypt/live/trust-ai-lab.eu/fullchain.pem

certificate_checksum() {
  if [ -r "$certificate" ]; then
    cksum "$certificate"
  fi
}

watch_certificate() {
  previous_checksum=$(certificate_checksum)

  while :; do
    sleep 60
    current_checksum=$(certificate_checksum)

    if [ -n "$current_checksum" ] && [ "$current_checksum" != "$previous_checksum" ]; then
      if nginx -t; then
        nginx -s reload
        previous_checksum=$current_checksum
        echo "Reloaded nginx after certificate update."
      else
        echo "Certificate changed, but nginx configuration validation failed; reload skipped." >&2
      fi
    fi
  done
}

if [ ! -r "$certificate" ]; then
  echo "Missing certificate: $certificate" >&2
  echo "Issue the initial certificate before starting the production override." >&2
  exit 1
fi

watch_certificate &
exec nginx -g "daemon off;"
