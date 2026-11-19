# PostgreSQL backups

The `db-backup` service creates a PostgreSQL custom-format dump at 02:00 on the
1st and 15th of each month, in the container's timezone (UTC by default). A
dump made on the first day of a month is also kept as a monthly backup (a hard
link, so it takes no extra space while the biweekly copy still exists).

Backups are written to `data/backups/postgres` with these default retention
limits:

- 1 biweekly backup (`biweekly/`, the latest dump)
- 12 monthly backups (`monthly/`)

A manual backup also counts as the latest biweekly dump, so it replaces the
previous one.

Each dump is written to a temporary location and checked with `pg_restore
--list` before it is published. A failed or interrupted dump is therefore not
mistaken for a valid backup.

## Configuration

Set any of these values in `.env` to override the defaults:

```dotenv
POSTGRES_BACKUP_CRON=0 2 1,15 * *
POSTGRES_BACKUP_ON_START=false
POSTGRES_BACKUP_KEEP_BIWEEKLY=1
POSTGRES_BACKUP_KEEP_MONTHLY=12
```

## Manual backup

```sh
docker compose run --rm db-backup backup-now
```

## Inspect a backup

```sh
docker compose run --rm --no-deps db-backup \
  pg_restore --list /backups/biweekly/NAME.dump
```

## Restore

Stop application services that write to the database before restoring. The
following command replaces objects in the target database, so select the dump
carefully:

```sh
docker compose exec -T db pg_restore \
  --clean --if-exists --no-owner \
  --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  < data/backups/postgres/biweekly/NAME.dump
```

A backup on the same host is not sufficient disaster recovery. Copy the
monthly directory to separate storage with appropriate access controls.
