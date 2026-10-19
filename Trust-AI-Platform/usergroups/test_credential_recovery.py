"""Recovering initial passwords encrypted with a key that has since changed."""
import io
from unittest.mock import patch

import openpyxl
from cryptography.fernet import Fernet
from django.contrib.auth.models import Group, User
from django.core.management import CommandError, call_command
from django.test import TestCase, override_settings
from django.urls import reverse

from usergroups.credentials import encrypt_initial_password, reveal_initial_password
from usergroups.models import UserGroup, UserGroupMembership

OLD_SECRET = 'old-secret-key-that-was-rotated'
NEW_SECRET = 'new-secret-key-after-rotation'
OLD_KEY = Fernet.generate_key().decode()
NEW_KEY = Fernet.generate_key().decode()


class RecoveryBase(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.teacher = User.objects.create_user('rec_teacher', password='pass')
        self.teacher.groups.add(teachers)
        self.group = UserGroup.objects.create(
            name='Marathon', prefix='mar', number_of_users=2, created_by=self.teacher,
        )

    def member(self, username, stored):
        student = User.objects.create_user(username)
        return UserGroupMembership.objects.create(
            group=self.group, user=student, initial_password=stored,
        )

    def recover(self, *answers, apply=False):
        """Run the command, answering its two hidden prompts in order."""
        out = io.StringIO()
        with patch(
            'usergroups.management.commands.recover_student_credentials.getpass',
            side_effect=list(answers),
        ):
            call_command('recover_student_credentials', *(['--apply'] if apply else []), stdout=out)
        return out.getvalue()


class RecoverWithOldSecretKeyTests(RecoveryBase):
    def setUp(self):
        super().setUp()
        with self.settings(SECRET_KEY=OLD_SECRET, STUDENT_CREDENTIALS_KEY=''):
            self.first = self.member('mar1', encrypt_initial_password('apple123'))
            self.second = self.member('mar2', encrypt_initial_password('pear4567'))

    @override_settings(SECRET_KEY=NEW_SECRET, STUDENT_CREDENTIALS_KEY=NEW_KEY)
    def test_rotated_secret_key_makes_passwords_unreadable(self):
        self.first.refresh_from_db()
        self.assertEqual(reveal_initial_password(self.first.initial_password), '')

    @override_settings(SECRET_KEY=NEW_SECRET, STUDENT_CREDENTIALS_KEY=NEW_KEY)
    def test_dry_run_reports_but_changes_nothing(self):
        before = self.first.initial_password

        output = self.recover(OLD_SECRET, '')

        self.assertIn('2 can be recovered', output)
        self.assertIn('--apply', output)
        self.first.refresh_from_db()
        self.assertEqual(self.first.initial_password, before)

    @override_settings(SECRET_KEY=NEW_SECRET, STUDENT_CREDENTIALS_KEY=NEW_KEY)
    def test_apply_reencrypts_with_the_current_key(self):
        output = self.recover(OLD_SECRET, '', apply=True)

        self.assertIn('Recovered 2', output)
        self.first.refresh_from_db()
        self.second.refresh_from_db()
        self.assertEqual(reveal_initial_password(self.first.initial_password), 'apple123')
        self.assertEqual(reveal_initial_password(self.second.initial_password), 'pear4567')

    @override_settings(SECRET_KEY=NEW_SECRET, STUDENT_CREDENTIALS_KEY=NEW_KEY)
    def test_wrong_old_key_recovers_nothing(self):
        output = self.recover('not-the-old-secret', '', apply=True)

        self.assertIn('0 can be recovered', output)
        self.assertIn('2 still cannot be read', output)
        self.first.refresh_from_db()
        self.assertEqual(reveal_initial_password(self.first.initial_password), '')

    @override_settings(SECRET_KEY=NEW_SECRET, STUDENT_CREDENTIALS_KEY=NEW_KEY)
    def test_no_old_key_given_is_refused(self):
        with self.assertRaises(CommandError):
            self.recover('', '')


class RecoverWithOldCredentialsKeyTests(RecoveryBase):
    @override_settings(SECRET_KEY=NEW_SECRET, STUDENT_CREDENTIALS_KEY=NEW_KEY)
    def test_old_credentials_key_recovers_and_readable_ones_are_kept(self):
        with self.settings(STUDENT_CREDENTIALS_KEY=OLD_KEY):
            old = self.member('mar1', encrypt_initial_password('apple123'))
        current = self.member('mar2', encrypt_initial_password('pear4567'))
        current_value = current.initial_password

        output = self.recover('', OLD_KEY, apply=True)

        self.assertIn('1 already readable', output)
        self.assertIn('Recovered 1', output)
        old.refresh_from_db()
        current.refresh_from_db()
        self.assertEqual(reveal_initial_password(old.initial_password), 'apple123')
        self.assertEqual(current.initial_password, current_value)

    @override_settings(SECRET_KEY=NEW_SECRET, STUDENT_CREDENTIALS_KEY=NEW_KEY)
    def test_malformed_credentials_key_is_refused(self):
        with self.assertRaises(CommandError):
            self.recover('', 'not-a-fernet-key')


@override_settings(SECRET_KEY=NEW_SECRET, STUDENT_CREDENTIALS_KEY=NEW_KEY)
class UnreadablePasswordDisplayTests(RecoveryBase):
    def setUp(self):
        super().setUp()
        with self.settings(SECRET_KEY=OLD_SECRET, STUDENT_CREDENTIALS_KEY=''):
            self.member('mar1', encrypt_initial_password('apple123'))
        self.member('mar2', encrypt_initial_password('pear4567'))
        self.client.force_login(self.teacher)

    def test_group_page_says_when_a_password_cannot_be_read(self):
        response = self.client.get(reverse('view_group', args=[self.group.id]))

        self.assertContains(response, "Can't be read")
        self.assertContains(response, 'pear4567')

    def test_excel_says_when_a_password_cannot_be_read(self):
        response = self.client.get(reverse('download_credentials', args=[self.group.id]))
        sheet = openpyxl.load_workbook(io.BytesIO(response.content)).active
        rows = {row[0]: row[1] for row in sheet.iter_rows(min_row=2, values_only=True)}

        self.assertEqual(rows['mar2'], 'pear4567')
        self.assertIn("can't be read", rows['mar1'])
