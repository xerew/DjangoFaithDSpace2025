from django.core.management.base import BaseCommand

from usergroups.credentials import reencrypt
from usergroups.models import UserGroupMembership


class Command(BaseCommand):
    help = (
        'Encrypt plaintext initial student passwords and re-encrypt all of '
        'them with the current STUDENT_CREDENTIALS_KEY. Run it after setting '
        'or changing that key, and before rotating SECRET_KEY.'
    )

    def handle(self, *args, **options):
        updated = 0
        for membership in UserGroupMembership.objects.exclude(initial_password=''):
            new_value = reencrypt(membership.initial_password)
            if new_value != membership.initial_password:
                membership.initial_password = new_value
                membership.save(update_fields=['initial_password'])
                updated += 1
        self.stdout.write(self.style.SUCCESS(f'Re-encrypted {updated} initial password(s).'))
