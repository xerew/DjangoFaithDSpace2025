"""Student accounts must not use passwords anyone can guess from the
(public) source code: <department>student<N>password123123."""
import io
import os
import tempfile

from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.core.management import call_command
from django.test import TestCase

from authoringtool.models import SchoolDepartment


class CreateStudentsTests(TestCase):
    def test_created_students_get_random_passwords_written_outside_the_repo(self):
        SchoolDepartment.objects.create(name='Physics')
        output = os.path.join(tempfile.mkdtemp(), 'credentials.txt')
        call_command('create_students', output=output, stdout=io.StringIO())

        with open(output, encoding='utf-8') as handle:
            lines = [line.strip() for line in handle if 'Password:' in line]
        self.assertEqual(len(lines), 25)
        username, password = lines[0].replace('Username: ', '').split(', Password: ')
        self.assertNotIn('password123123', password)
        self.assertIsNotNone(authenticate(username=username, password=password))


class LockGuessablePasswordsTests(TestCase):
    def setUp(self):
        self.guessable = User.objects.create_user(
            'physicsstudent1', password='physicsstudent1password123123',
        )
        self.safe = User.objects.create_user('physicsstudent2', password='a-random-secret-9f3k')

    def run_command(self, *args):
        out = io.StringIO()
        call_command('lock_guessable_student_passwords', *args, stdout=out)
        return out.getvalue()

    def test_dry_run_reports_and_changes_nothing(self):
        output = self.run_command()
        self.assertIn('1 account', output)
        self.guessable.refresh_from_db()
        self.assertTrue(self.guessable.check_password('physicsstudent1password123123'))

    def test_apply_locks_only_guessable_accounts_and_keeps_them(self):
        self.run_command('--apply')
        self.guessable.refresh_from_db()
        self.safe.refresh_from_db()
        self.assertFalse(self.guessable.has_usable_password())
        self.assertTrue(User.objects.filter(pk=self.guessable.pk).exists())
        self.assertTrue(self.safe.check_password('a-random-secret-9f3k'))
