"""
Test-only settings override.
Swaps PostgreSQL for an in-memory SQLite database and replaces the
Redis cache with the dummy backend so tests run without external services.
"""
from .settings import *  # noqa: F401, F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.dummy.DummyCache",
    }
}

# Silence Celery so no broker connection is attempted during tests
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True
SCENARIO_SIMILARITY_EMBEDDINGS_ENABLED = False

# Patch django.contrib.postgres range fields so their SQL placeholder degrades
# gracefully to plain %s on SQLite.  Without this, IntegerRangeField generates
# NULL::int4range in INSERT statements, which SQLite cannot parse.
from django.contrib.postgres.fields import IntegerRangeField as _IRF  # noqa: E402

_orig_placeholder = _IRF.get_placeholder


def _sqlite_safe_placeholder(self, value, compiler, connection):
    if connection.vendor == 'sqlite':
        return '%s'
    return _orig_placeholder(self, value, compiler, connection)


_IRF.get_placeholder = _sqlite_safe_placeholder
