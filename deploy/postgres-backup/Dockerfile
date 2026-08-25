FROM postgres:16-alpine

COPY backup-postgres.sh /usr/local/bin/backup-postgres
COPY entrypoint.sh /usr/local/bin/postgres-backup-entrypoint

RUN sed -i 's/\r$//' \
      /usr/local/bin/backup-postgres \
      /usr/local/bin/postgres-backup-entrypoint \
    && chmod 0755 \
      /usr/local/bin/backup-postgres \
      /usr/local/bin/postgres-backup-entrypoint

ENTRYPOINT ["/usr/local/bin/postgres-backup-entrypoint"]
