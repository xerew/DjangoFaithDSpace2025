from django.apps import AppConfig
from django.conf import settings
from django.core import checks


class UsergroupsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'usergroups'

    def ready(self):
        checks.register(student_credentials_key_check)


def student_credentials_key_check(app_configs, **kwargs):
    if settings.DEBUG or settings.STUDENT_CREDENTIALS_KEY:
        return []
    return [checks.Warning(
        'STUDENT_CREDENTIALS_KEY is not set, so initial student passwords are '
        'encrypted with a key derived from SECRET_KEY and become unreadable if '
        'SECRET_KEY changes.',
        hint='Set STUDENT_CREDENTIALS_KEY, then run "manage.py '
             'reencrypt_student_credentials" before rotating SECRET_KEY.',
        id='usergroups.W001',
    )]
