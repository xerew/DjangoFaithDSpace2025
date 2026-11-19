#!/bin/sh
set -eu

: "${PGHOST:?PGHOST is required}"
: "${PGDATABASE:?PGDATABASE is required}"
: "${PGUSER:?PGUSER is required}"

keep_biweekly=${BACKUP_KEEP_BIWEEKLY:-1}
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

require_positive_integer BACKUP_KEEP_BIWEEKLY "$keep_biweekly"
require_positive_integer BACKUP_KEEP_MONTHLY "$keep_monthly"

umask 077
mkdir -p /backups/biweekly /backups/monthly /backups/.in-progress

timestamp=$(date -u '+%Y-%m-%dT%H%M%SZ')
day_of_month=$(printf '%s' "$timestamp" | cut -c9-10)
safe_database=$(printf '%s' "$PGDATABASE" | tr -c 'A-Za-z0-9_.-' '_')
filename="${safe_database}_${timestamp}.dump"
temporary_file="/backups/.in-progress/${filename}.tmp"
biweekly_file="/backups/biweekly/$filename"

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
mv "$temporary_file" "$biweekly_file"
echo "Created verified biweekly backup: $biweekly_file"

if [ "$day_of_month" = '01' ]; then
  # A hard link takes no extra disk space while the biweekly copy still exists.
  ln "$biweekly_file" "/backups/monthly/$filename" 2>/dev/null \
    || cp "$biweekly_file" "/backups/monthly/$filename"
  echo "Created monthly backup: /backups/monthly/$filename"
fi

prune_directory /backups/biweekly "$keep_biweekly"
prune_directory /backups/monthly "$keep_monthly"

trap - EXIT INT TERM
echo "PostgreSQL backup completed successfully."
