from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User, Group
from authoringtool.models import Scenario
from usergroups.models import UserGroup


def make_teacher(username='teacher1', password='pass'):
    user = User.objects.create_user(username, password=password)
    group = Group.objects.get_or_create(name='teachers')[0]
    user.groups.add(group)
    return user


class TeacherHomeAccessTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.teacher = make_teacher()
        self.regular = User.objects.create_user('regular', password='pass')

    def test_anonymous_redirected(self):
        r = self.client.get(reverse('teacher_home'))
        self.assertEqual(r.status_code, 302)
        self.assertIn('/login', r['Location'])

    def test_non_teacher_forbidden(self):
        self.client.login(username='regular', password='pass')
        r = self.client.get(reverse('teacher_home'))
        self.assertEqual(r.status_code, 403)

    def test_teacher_gets_200(self):
        self.client.login(username='teacher1', password='pass')
        r = self.client.get(reverse('teacher_home'))
        self.assertEqual(r.status_code, 200)


class TeacherHomeContextTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.teacher = make_teacher()
        self.client.login(username='teacher1', password='pass')

    def test_context_has_stat_keys(self):
        r = self.client.get(reverse('teacher_home'))
        for key in ('my_scenario_count', 'my_group_count', 'total_students', 'latest_public', 'show_get_started'):
            self.assertIn(key, r.context, msg=f'Missing context key: {key}')

    def test_show_get_started_true_when_no_scenarios(self):
        r = self.client.get(reverse('teacher_home'))
        self.assertTrue(r.context['show_get_started'])

    def test_show_get_started_false_when_has_scenario(self):
        Scenario.objects.create(
            name='My Scenario', created_by=self.teacher, updated_by=self.teacher
        )
        r = self.client.get(reverse('teacher_home'))
        self.assertFalse(r.context['show_get_started'])

    def test_my_scenario_count(self):
        Scenario.objects.create(name='S1', created_by=self.teacher, updated_by=self.teacher)
        Scenario.objects.create(name='S2', created_by=self.teacher, updated_by=self.teacher)
        r = self.client.get(reverse('teacher_home'))
        self.assertEqual(r.context['my_scenario_count'], 2)

    def test_latest_public_max_5(self):
        for i in range(7):
            Scenario.objects.create(
                name=f'Pub{i}', created_by=self.teacher, updated_by=self.teacher,
                visibility_status='public'
            )
        r = self.client.get(reverse('teacher_home'))
        self.assertLessEqual(len(r.context['latest_public']), 5)

    def test_latest_public_only_public(self):
        Scenario.objects.create(
            name='Private', created_by=self.teacher, updated_by=self.teacher,
            visibility_status='private'
        )
        r = self.client.get(reverse('teacher_home'))
        self.assertEqual(len(r.context['latest_public']), 0)
