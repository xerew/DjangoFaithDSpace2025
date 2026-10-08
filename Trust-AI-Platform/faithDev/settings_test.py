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
# Keep the revision-draft workflow under test; tests of the switched-off
# behaviour override this.
SCENARIO_REVISION_PROTECTION = True

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

# ArrayField (QuestionBunch.activity_ids) casts its placeholder to a Postgres
# array type; on SQLite store the list as JSON text instead.
import json as _json  # noqa: E402

from django.contrib.postgres.fields import ArrayField as _AF  # noqa: E402

_orig_array_placeholder = _AF.get_placeholder
_orig_array_prep = _AF.get_db_prep_value


def _sqlite_array_placeholder(self, value, compiler, connection):
    if connection.vendor == 'sqlite':
        return '%s'
    return _orig_array_placeholder(self, value, compiler, connection)


def _sqlite_array_prep(self, value, connection, prepared=False):
    if connection.vendor == 'sqlite' and isinstance(value, (list, tuple)):
        return _json.dumps(list(value))
    return _orig_array_prep(self, value, connection, prepared)


def _sqlite_array_from_db(self, value, expression, connection):
    if connection.vendor == 'sqlite' and isinstance(value, str):
        return _json.loads(value)
    return value


_AF.get_placeholder = _sqlite_array_placeholder
_AF.get_db_prep_value = _sqlite_array_prep
_AF.from_db_value = _sqlite_array_from_db
