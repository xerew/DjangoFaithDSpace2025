from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = (
        'Find student accounts whose password is the old guessable pattern '
        '<username>password123123 and, with --apply, make that password '
        'unusable. The accounts and all their work are kept; an admin can set '
        'a new password when a student needs to log in again.'
    )

    def add_arguments(self, parser):
        parser.add_argument('--apply', action='store_true',
                            help='Lock the accounts (default: only report).')

    def handle(self, *args, apply=False, **options):
        guessable = [
            user for user in User.objects.filter(username__icontains='student')
            if user.has_usable_password()
            and user.check_password(f'{user.username}password123123')
        ]
        if apply:
            for user in guessable:
                user.set_unusable_password()
                user.save(update_fields=['password'])
            self.stdout.write(self.style.SUCCESS(
                f'Locked {len(guessable)} account(s) with a guessable password.'
            ))
        else:
            self.stdout.write(
                f'{len(guessable)} account(s) use the guessable password pattern. '
                'Run again with --apply to lock them.'
            )
