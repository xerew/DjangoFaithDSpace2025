from django.apps import AppConfig


class AuthoringtoolConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'authoringtool'

    def ready(self):
        import authoringtool.signals
