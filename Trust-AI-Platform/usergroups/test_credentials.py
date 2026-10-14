import io

import openpyxl
from django.contrib.auth import authenticate
from django.contrib.auth.models import Group, User
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse

from usergroups.credentials import ENCRYPTED_PREFIX, reveal_initial_password
from usergroups.models import UserGroup, UserGroupMembership


class InitialPasswordEncryptionTests(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.teacher = User.objects.create_user('cred_teacher', password='pass')
        self.teacher.groups.add(teachers)
        self.client.force_login(self.teacher)

    def create_group(self):
        self.client.post(reverse('create_group'), {
            'name': 'Class B', 'prefix': 'clb', 'number_of_users': 2,
        })
        return UserGroup.objects.get(name='Class B')

    def downloaded(self, group):
        response = self.client.get(reverse('download_credentials', args=[group.id]))
        sheet = openpyxl.load_workbook(io.BytesIO(response.content)).active
        return {row[0]: row[1] for row in sheet.iter_rows(min_row=2, values_only=True)}

    def test_passwords_are_stored_encrypted(self):
        group = self.create_group()
        for membership in UserGroupMembership.objects.filter(group=group):
            self.assertTrue(membership.initial_password.startswith(ENCRYPTED_PREFIX))

    def test_teachers_still_download_working_passwords(self):
        group = self.create_group()
        credentials = self.downloaded(group)
        self.assertEqual(len(credentials), 2)
        for username, password in credentials.items():
            self.assertFalse(password.startswith(ENCRYPTED_PREFIX))
            self.assertIsNotNone(authenticate(username=username, password=password))

    def test_group_page_shows_readable_passwords(self):
        group = self.create_group()
        response = self.client.get(reverse('view_group', args=[group.id]))
        shown = {m.user.username: m.password_plain for m in response.context['memberships']}
        self.assertEqual(shown, self.downloaded(group))

    def test_legacy_plaintext_is_read_and_can_be_reencrypted(self):
        group = UserGroup.objects.create(
            name='Old class', prefix='old', number_of_users=1, created_by=self.teacher,
        )
        student = User.objects.create_user('old1', password='legacy123')
        membership = UserGroupMembership.objects.create(
            group=group, user=student, initial_password='legacy123',
        )
        self.assertEqual(reveal_initial_password(membership.initial_password), 'legacy123')
        call_command('reencrypt_student_credentials', stdout=io.StringIO())
        membership.refresh_from_db()
        self.assertTrue(membership.initial_password.startswith(ENCRYPTED_PREFIX))
        self.assertEqual(self.downloaded(group), {'old1': 'legacy123'})
