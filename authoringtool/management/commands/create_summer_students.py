import string
import secrets
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from authoringtool.models import SchoolDepartment

class Command(BaseCommand):
    help = 'Create 50 users for the Balkan Summer School department with randomized passwords'

    def handle(self, *args, **kwargs):
        # Ensure the "Summer School" department exists
        summer_school, created = SchoolDepartment.objects.get_or_create(name='Balkan School')
        if created:
            self.stdout.write(self.style.SUCCESS('Department "Balkan School" created successfully'))
        else:
            self.stdout.write(self.style.WARNING('Department "Balkan School" already exists'))

        # Function to generate a random password
        def generate_random_password(length=12):
            characters = string.ascii_letters + string.digits
            password = ''.join(secrets.choice(characters) for _ in range(length))
            return password

        # Create or update 10 users for the "Balkan School" department
        for i in range(1, 51):
            username = f'BalkanGrStudent{i}'
            password = generate_random_password()
            email = f'{username}@example.com'
            
            user, created = User.objects.get_or_create(username=username, defaults={
                'email': email,
                'password': password,
            })

            if created:
                user.set_password(password)  # Set the randomized password
                user.school_department = summer_school
                user.save()
                self.stdout.write(self.style.SUCCESS(f'User {username} created successfully with password {password}'))
            else:
                user.set_password(password)  # Update the password for existing user
                user.save()
                self.stdout.write(self.style.SUCCESS(f'User {username} password updated to {password}'))
