from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from authoringtool.models import SchoolDepartment

class Command(BaseCommand):
    help = 'Create 25 users for each school department and output their credentials to a file'

    def handle(self, *args, **kwargs):
        departments = SchoolDepartment.objects.all()
        with open('user_credentials.txt', 'w') as file:
            for department in departments:
                for i in range(1, 26):
                    username = f'{department.name.lower()}student{i}'
                    password = f'{department.name.lower()}student{i}password123123'
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
        self.stdout.write(self.style.SUCCESS('User creation and credential logging completed'))
