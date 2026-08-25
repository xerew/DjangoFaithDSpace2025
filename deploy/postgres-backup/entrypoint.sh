#!/bin/sh
set -eu

wait_for_postgres() {
  echo "Waiting for PostgreSQL at ${PGHOST}:${PGPORT:-5432}."
  until pg_isready \
    --host "$PGHOST" \
    --port "${PGPORT:-5432}" \
    --username "$PGUSER" \
    --dbname "$PGDATABASE" >/dev/null 2>&1; do
    sleep 2
  done
}

if [ "${1:-}" = 'backup-now' ]; then
  wait_for_postgres
  exec /usr/local/bin/backup-postgres
fi

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

wait_for_postgres

if [ "${BACKUP_ON_START:-false}" = 'true' ]; then
  /usr/local/bin/backup-postgres
fi

backup_cron=${BACKUP_CRON:-0 2 * * *}
printf '%s %s\n' \
  "$backup_cron" \
  '/usr/local/bin/backup-postgres >>/proc/1/fd/1 2>>/proc/1/fd/2' \
  > /etc/crontabs/root

echo "PostgreSQL backup scheduler started: $backup_cron (container time)."
exec crond -f -l 2
