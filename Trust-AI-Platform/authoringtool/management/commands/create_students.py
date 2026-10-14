import secrets

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from authoringtool.models import SchoolDepartment


class Command(BaseCommand):
    help = (
        'Create 25 users for each school department with random passwords and '
        'write their credentials to --output (keep that file out of the repository).'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--output', required=True,
            help='Where to write the credentials, e.g. /root/credentials.txt',
        )

    def handle(self, *args, output, **kwargs):
        departments = SchoolDepartment.objects.all()
        with open(output, 'w', encoding='utf-8') as file:
            for department in departments:
                for i in range(1, 26):
                    username = f'{department.name.lower()}student{i}'
                    password = secrets.token_urlsafe(9)
                    email = f'{username}@example.com'
                    if not User.objects.filter(username=username).exists():
                        user = User.objects.create_user(
                            username=username,
                            password=password,
                            email=email,
                        )
                        user.school_department = department
                        user.save()
                        self.stdout.write(self.style.SUCCESS(f'User {username} created successfully'))
                        file.write(f'Username: {username}, Password: {password}\n')
                    else:
                        self.stdout.write(self.style.WARNING(f'User {username} already exists'))
                        file.write(f'Username: {username} already exists\n')
        self.stdout.write(self.style.SUCCESS(f'Credentials written to {output}'))
