from django.contrib.auth.models import Group, User
from django.db.models import ProtectedError
from django.test import TestCase
from django.urls import reverse

from authoringtool.models import Activity, ActivityType, Phase, Scenario, UserAnswer
from usergroups.models import UserGroup, UserGroupMembership


class StudentDataMixin:
    def make_teacher(self, username, staff=False):
        teachers, _ = Group.objects.get_or_create(name='teachers')
        user = User.objects.create_user(username, password='pass', is_staff=staff)
        user.groups.add(teachers)
        return user

    def make_group(self, owner, size=2):
        group = UserGroup.objects.create(
            name=f'{owner.username} class', prefix=f'{owner.username[:4]}s',
            number_of_users=size, created_by=owner,
        )
        students = []
        for i in range(size):
            student = User.objects.create_user(f'{group.prefix}{i + 1}', password='pass')
            UserGroupMembership.objects.create(group=group, user=student)
            students.append(student)
        return group, students

    def give_answer(self, student, owner):
        scenario, _ = Scenario.objects.get_or_create(
            name='Retention scenario',
            defaults={'created_by': owner, 'updated_by': owner},
        )
        phase, _ = Phase.objects.get_or_create(
            name='Phase', scenario=scenario,
            defaults={'created_by': owner, 'updated_by': owner},
        )
        activity_type, _ = ActivityType.objects.get_or_create(
            name='Explanation', defaults={'created_by': owner, 'updated_by': owner},
        )
        activity, _ = Activity.objects.get_or_create(
            name='Read', scenario=scenario, phase=phase,
            defaults={
                'text': 'x', 'activity_type': activity_type,
                'created_by': owner, 'updated_by': owner,
            },
        )
        return UserAnswer.objects.create(user=student, activity=activity, timing=20)


class GroupArchiveTests(StudentDataMixin, TestCase):
    def setUp(self):
        self.teacher = self.make_teacher('arch_teacher')
        self.group, self.students = self.make_group(self.teacher)
        self.answer = self.give_answer(self.students[0], self.teacher)
        self.client.force_login(self.teacher)

    def test_delete_archives_group_and_keeps_students_and_their_work(self):
        response = self.client.post(reverse('delete_group', args=[self.group.id]))
        self.assertRedirects(response, reverse('list_groups'))
        self.group.refresh_from_db()
        self.assertIsNotNone(self.group.archived_at)
        self.assertEqual(self.group.archived_by, self.teacher)
        for student in self.students:
            student.refresh_from_db()
            self.assertFalse(student.is_active, 'archived students cannot log in')
        self.assertTrue(UserAnswer.objects.filter(pk=self.answer.pk).exists())

    def test_archived_student_cannot_log_in(self):
        self.client.post(reverse('delete_group', args=[self.group.id]))
        self.client.logout()
        self.assertFalse(self.client.login(username=self.students[0].username, password='pass'))

    def test_get_does_not_archive(self):
        self.client.get(reverse('delete_group', args=[self.group.id]))
        self.group.refresh_from_db()
        self.assertIsNone(self.group.archived_at)

    def test_archived_groups_are_listed_separately(self):
        self.client.post(reverse('delete_group', args=[self.group.id]))
        response = self.client.get(reverse('list_groups'))
        self.assertNotIn(self.group, list(response.context['groups']))
        self.assertIn(self.group, list(response.context['archived_groups']))

    def test_restore_brings_group_and_logins_back(self):
        self.client.post(reverse('delete_group', args=[self.group.id]))
        response = self.client.post(reverse('restore_group', args=[self.group.id]))
        self.assertRedirects(response, reverse('list_groups'))
        self.group.refresh_from_db()
        self.assertIsNone(self.group.archived_at)
        for student in self.students:
            student.refresh_from_db()
            self.assertTrue(student.is_active)

    def test_other_teacher_cannot_archive_or_restore(self):
        other = self.make_teacher('arch_other')
        self.client.force_login(other)
        self.assertEqual(
            self.client.post(reverse('delete_group', args=[self.group.id])).status_code, 404,
        )
        self.assertEqual(
            self.client.post(reverse('restore_group', args=[self.group.id])).status_code, 404,
        )

    def test_admin_can_restore_any_group(self):
        self.client.post(reverse('delete_group', args=[self.group.id]))
        self.client.force_login(self.make_teacher('arch_admin', staff=True))
        self.client.post(reverse('restore_group', args=[self.group.id]))
        self.group.refresh_from_db()
        self.assertIsNone(self.group.archived_at)


class ShrinkGroupTests(StudentDataMixin, TestCase):
    def setUp(self):
        self.teacher = self.make_teacher('shrink_teacher')
        self.group, self.students = self.make_group(self.teacher, size=3)
        self.client.force_login(self.teacher)

    def shrink_to(self, size):
        return self.client.post(reverse('edit_group', args=[self.group.id]), {
            'name': self.group.name, 'prefix': self.group.prefix,
            'number_of_users': size,
        })

    def test_removed_students_with_work_are_deactivated_not_deleted(self):
        last = self.students[-1]
        answer = self.give_answer(last, self.teacher)
        self.shrink_to(2)
        last.refresh_from_db()
        self.assertFalse(last.is_active)
        self.assertTrue(UserAnswer.objects.filter(pk=answer.pk).exists())

    def test_removed_students_without_work_are_deleted(self):
        last = self.students[-1]
        self.shrink_to(2)
        self.assertFalse(User.objects.filter(pk=last.pk).exists())


class StudentDataSafetyNetTests(StudentDataMixin, TestCase):
    def setUp(self):
        self.teacher = self.make_teacher('net_teacher')
        _, students = self.make_group(self.teacher, size=1)
        self.student = students[0]
        self.give_answer(self.student, self.teacher)

    def test_database_refuses_to_delete_a_student_with_answers(self):
        with self.assertRaises(ProtectedError):
            self.student.delete()

    def test_admin_delete_user_refuses_and_suggests_suspending(self):
        admin = self.make_teacher('net_admin', staff=True)
        self.client.force_login(admin)
        response = self.client.post(reverse('admin_delete_user', args=[self.student.id]))
        self.assertFalse(response.json()['success'])
        self.assertIn('Suspend', response.json()['error'])
        self.assertTrue(User.objects.filter(pk=self.student.pk).exists())
