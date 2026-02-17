import csv
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User

class Command(BaseCommand):
    help = 'Export user data to CSV'

    def handle(self, *args, **kwargs):
        users = User.objects.all()

        # Specify the file path for the export
        file_path = '/home/ec2-user/users_export.csv'

        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['username', 'password', 'email', 'school_department']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for user in users:
                writer.writerow({
                    'username': user.username,
                    'password': user.username+'password123123',
                    'email': user.email,
                    'school_department': user.school_department.name if user.school_department else 'N/A'
                })

        self.stdout.write(self.style.SUCCESS(f'Successfully exported user data to {file_path}'))