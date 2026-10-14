from unittest.mock import patch

from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.models import Activity, ActivityType, Phase, Scenario
from usergroups.models import UserGroup, UserGroupMembership


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class StudentScenarioAccessTests(TestCase):
    """Students may only open scenarios assigned to one of their groups."""

    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('access_owner', password='pass')
        self.other_teacher = User.objects.create_user('access_other', password='pass')
        for teacher in (self.owner, self.other_teacher):
            teacher.groups.add(teachers)
        self.student = User.objects.create_user('access_student', password='pass')

        self.assigned = self.scenario('Assigned scenario', 'private')
        self.unassigned = self.scenario('Unassigned public scenario', 'public')
        group = UserGroup.objects.create(
            name='Class A', prefix='cla', number_of_users=1, created_by=self.owner,
        )
        UserGroupMembership.objects.create(group=group, user=self.student)
        group.assigned_scenarios.add(self.assigned)

    def scenario(self, name, visibility):
        scenario = Scenario.objects.create(
            name=name, visibility_status=visibility,
            created_by=self.owner, updated_by=self.owner,
        )
        phase = Phase.objects.create(
            name='Phase', scenario=scenario,
            created_by=self.owner, updated_by=self.owner,
        )
        activity_type, _ = ActivityType.objects.get_or_create(
            name='Explanation',
            defaults={'created_by': self.owner, 'updated_by': self.owner},
        )
        scenario.first_activity = Activity.objects.create(
            name=f'{name} intro', text='<p>Hello</p>', phase=phase,
            scenario=scenario, activity_type=activity_type,
            created_by=self.owner, updated_by=self.owner,
        )
        return scenario

    def test_student_opens_assigned_scenario(self):
        self.client.force_login(self.student)
        response = self.client.get(reverse('studentView', args=[self.assigned.id]))
        self.assertEqual(response.status_code, 200)

    def test_student_cannot_open_unassigned_scenario(self):
        self.client.force_login(self.student)
        response = self.client.get(reverse('studentView', args=[self.unassigned.id]))
        self.assertEqual(response.status_code, 404)

    def test_student_cannot_fetch_activity_of_unassigned_scenario(self):
        self.client.force_login(self.student)
        activity = self.unassigned.first_activity
        response = self.client.get(reverse('get_activity', args=[activity.id]))
        self.assertEqual(response.status_code, 404)

    def test_student_fetches_activity_of_assigned_scenario(self):
        self.client.force_login(self.student)
        activity = self.assigned.first_activity
        response = self.client.get(reverse('get_activity', args=[activity.id]))
        self.assertEqual(response.status_code, 200)

    def test_student_cannot_open_questionnaire_of_unassigned_scenario(self):
        self.client.force_login(self.student)
        response = self.client.get(
            reverse('scenario_questions', args=[self.unassigned.id]), follow=True,
        )
        self.assertEqual(response.status_code, 404)

    def test_student_cannot_submit_questionnaire_of_unassigned_scenario(self):
        self.client.force_login(self.student)
        response = self.client.post(
            reverse('submit_answers', args=[self.unassigned.id]), {},
        )
        self.assertEqual(response.status_code, 404)

    def test_student_cannot_pause_unassigned_scenario(self):
        self.client.force_login(self.student)
        response = self.client.post(
            reverse('help_start', args=[self.unassigned.id]),
            {'activity_id': self.unassigned.first_activity.id},
        )
        self.assertEqual(response.status_code, 404)

    def test_teacher_cannot_preview_private_scenario_of_someone_else(self):
        self.client.force_login(self.other_teacher)
        response = self.client.get(reverse('studentView', args=[self.assigned.id]))
        self.assertEqual(response.status_code, 403)

    def test_teacher_previews_public_scenario(self):
        self.client.force_login(self.other_teacher)
        response = self.client.get(reverse('studentView', args=[self.unassigned.id]))
        self.assertEqual(response.status_code, 200)


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class TeacherPageAccessTests(TestCase):
    """Teacher-only pages need a teacher who can view the scenario."""

    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('pages_owner', password='pass')
        self.other_teacher = User.objects.create_user('pages_other', password='pass')
        for teacher in (self.owner, self.other_teacher):
            teacher.groups.add(teachers)
        self.student = User.objects.create_user('pages_student', password='pass')
        self.private = Scenario.objects.create(
            name='Private pages', visibility_status='private',
            created_by=self.owner, updated_by=self.owner,
        )
        self.phase = Phase.objects.create(
            name='Phase', scenario=self.private,
            created_by=self.owner, updated_by=self.owner,
        )
        activity_type = ActivityType.objects.create(
            name='Explanation', created_by=self.owner, updated_by=self.owner,
        )
        self.activity = Activity.objects.create(
            name='Intro', text='<p>Hello</p>', phase=self.phase,
            scenario=self.private, activity_type=activity_type,
            created_by=self.owner, updated_by=self.owner,
        )

    def urls(self):
        return {
            'phase': reverse('phase', args=[self.private.id, self.phase.id]),
            'activity': reverse(
                'activity', args=[self.private.id, self.phase.id, self.activity.id]
            ),
            'export_scenario': reverse('export_scenario', args=[self.private.id]),
        }

    def test_students_cannot_open_teacher_pages(self):
        self.client.force_login(self.student)
        for name, url in self.urls().items():
            with self.subTest(page=name):
                self.assertEqual(self.client.get(url).status_code, 403)

    def test_other_teacher_cannot_open_private_scenario_pages(self):
        self.client.force_login(self.other_teacher)
        for name, url in self.urls().items():
            with self.subTest(page=name):
                self.assertEqual(self.client.get(url).status_code, 403)

    def test_owner_opens_own_pages(self):
        self.client.force_login(self.owner)
        with patch('authoringtool.exporter.ScenarioExporter.to_zip_bytes', return_value=b'zip'):
            for name, url in self.urls().items():
                with self.subTest(page=name):
                    self.assertEqual(self.client.get(url).status_code, 200)
