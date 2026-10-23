from django.core.management import call_command
from django.core.management.base import BaseCommand

from localization.languages import LANGUAGE_META

IGNORE = ['staticfiles/*', 'static/*', 'docs/*', 'media/*',
          'ai_metrics_cache/*', '*/migrations/*']


class Command(BaseCommand):
    help = 'Extract translatable strings for the 12 non-English languages and compile them.'

    def handle(self, *args, **options):
        codes = [m['code'] for m in LANGUAGE_META if m['code'] != 'en']
        verbosity = options.get('verbosity', 1)
        call_command('makemessages', locale=codes, ignore_patterns=IGNORE,
                     no_obsolete=True, add_location='file', verbosity=verbosity)
        call_command('compilemessages', locale=codes, verbosity=verbosity)
