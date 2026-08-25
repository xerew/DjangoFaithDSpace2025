# PostgreSQL backups

The `db-backup` service creates a PostgreSQL custom-format dump every day at
02:00 in the container's timezone (UTC by default). A Sunday dump is also kept
as a weekly backup, and a dump made on the first day of a month is also kept as
a monthly backup.

Backups are written to `data/backups/postgres` with these default retention
limits:

- 7 daily backups
- 5 weekly backups
- 12 monthly backups

Each dump is written to a temporary location and checked with `pg_restore
--list` before it is published. A failed or interrupted dump is therefore not
mistaken for a valid backup.

## Configuration

Set any of these values in `.env` to override the defaults:

```dotenv
POSTGRES_BACKUP_CRON=0 2 * * *
POSTGRES_BACKUP_ON_START=false
POSTGRES_BACKUP_KEEP_DAILY=7
POSTGRES_BACKUP_KEEP_WEEKLY=5
POSTGRES_BACKUP_KEEP_MONTHLY=12
```

## Manual backup

```sh
docker compose run --rm db-backup backup-now
```

## Inspect a backup

```sh
docker compose run --rm --no-deps db-backup \
  pg_restore --list /backups/daily/NAME.dump
```

## Restore

Stop application services that write to the database before restoring. The
following command replaces objects in the target database, so select the dump
carefully:

```sh
docker compose exec -T db pg_restore \
  --clean --if-exists --no-owner \
  --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  < data/backups/postgres/daily/NAME.dump
```

A backup on the same host is not sufficient disaster recovery. Copy the
monthly directory to separate storage with appropriate access controls.
