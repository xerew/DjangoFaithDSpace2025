#!/bin/sh
set -eu

: "${PGHOST:?PGHOST is required}"
: "${PGDATABASE:?PGDATABASE is required}"
: "${PGUSER:?PGUSER is required}"

keep_daily=${BACKUP_KEEP_DAILY:-7}
keep_weekly=${BACKUP_KEEP_WEEKLY:-5}
keep_monthly=${BACKUP_KEEP_MONTHLY:-12}

require_positive_integer() {
  name=$1
  value=$2
  case "$value" in
    ''|*[!0-9]*|0)
      echo "$name must be a positive integer; received '$value'." >&2
      exit 1
      ;;
  esac
}

prune_directory() {
  directory=$1
  keep=$2
  count=0

  find "$directory" -maxdepth 1 -type f -name '*.dump' -print \
    | sort -r \
    | while IFS= read -r backup_file; do
        count=$((count + 1))
        if [ "$count" -gt "$keep" ]; then
          rm -f -- "$backup_file"
          echo "Removed expired backup: $backup_file"
        fi
      done
}

require_positive_integer BACKUP_KEEP_DAILY "$keep_daily"
require_positive_integer BACKUP_KEEP_WEEKLY "$keep_weekly"
require_positive_integer BACKUP_KEEP_MONTHLY "$keep_monthly"

umask 077
mkdir -p /backups/daily /backups/weekly /backups/monthly /backups/.in-progress

timestamp=$(date -u '+%Y-%m-%dT%H%M%SZ')
safe_database=$(printf '%s' "$PGDATABASE" | tr -c 'A-Za-z0-9_.-' '_')
filename="${safe_database}_${timestamp}.dump"
temporary_file="/backups/.in-progress/${filename}.tmp"
daily_file="/backups/daily/$filename"

cleanup() {
  rm -f -- "$temporary_file"
}
trap cleanup EXIT INT TERM

echo "Starting PostgreSQL backup for '$PGDATABASE' at $timestamp."
pg_dump \
  --format=custom \
  --compress=6 \
  --no-owner \
  --no-privileges \
  --file "$temporary_file" \
  "$PGDATABASE"

# A dump is only published after pg_restore can read its archive catalogue.
pg_restore --list "$temporary_file" >/dev/null
mv "$temporary_file" "$daily_file"
echo "Created verified daily backup: $daily_file"

if [ "$(date -u '+%u')" = '7' ]; then
  cp "$daily_file" "/backups/weekly/$filename"
  echo "Created weekly backup: /backups/weekly/$filename"
fi

if [ "$(date -u '+%d')" = '01' ]; then
  cp "$daily_file" "/backups/monthly/$filename"
  echo "Created monthly backup: /backups/monthly/$filename"
fi

prune_directory /backups/daily "$keep_daily"
prune_directory /backups/weekly "$keep_weekly"
prune_directory /backups/monthly "$keep_monthly"

trap - EXIT INT TERM
echo "PostgreSQL backup completed successfully."
