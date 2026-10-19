from getpass import getpass

from django.core.management.base import BaseCommand, CommandError

from usergroups.credentials import old_key_fernet, recover
from usergroups.models import UserGroupMembership


class Command(BaseCommand):
    help = (
        'Recover initial student passwords that were encrypted with a key that '
        'has since changed (an old SECRET_KEY or STUDENT_CREDENTIALS_KEY) and '
        're-encrypt them with the current key. The old keys are asked for at a '
        'hidden prompt and never stored. Reports only, unless --apply is given.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--apply', action='store_true',
            help='Save the recovered passwords (re-encrypted with the current key).',
        )

    def handle(self, *args, **options):
        old_secret = getpass('Old DJANGO_SECRET_KEY (leave blank to skip): ').strip()
        old_key = getpass('Old STUDENT_CREDENTIALS_KEY (leave blank to skip): ').strip()
        if not old_secret and not old_key:
            raise CommandError('Give at least one old key.')
        try:
            fernet = old_key_fernet(
                secret_keys=[old_secret] if old_secret else [],
                credential_keys=[old_key] if old_key else [],
            )
        except ValueError:
            raise CommandError('The old STUDENT_CREDENTIALS_KEY is not a valid key.')

        counts = {'readable': 0, 'recovered': 0, 'unreadable': 0}
        recovered = []
        for membership in UserGroupMembership.objects.exclude(initial_password=''):
            state, value = recover(membership.initial_password, fernet)
            counts[state] += 1
            if state == 'recovered':
                membership.initial_password = value
                recovered.append(membership)

        self.stdout.write(
            f"{counts['readable']} already readable, "
            f"{counts['recovered']} can be recovered, "
            f"{counts['unreadable']} still cannot be read."
        )
        if not options['apply']:
            if recovered:
                self.stdout.write('Nothing saved. Run again with --apply to save them.')
            return
        UserGroupMembership.objects.bulk_update(recovered, ['initial_password'], batch_size=500)
        self.stdout.write(self.style.SUCCESS(f'Recovered {len(recovered)} initial password(s).'))
