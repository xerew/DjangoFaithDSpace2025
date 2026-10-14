import csv
import io
import uuid
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import Group, User
from django.test import TestCase
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    Phase,
    Scenario,
    SchoolDepartment,
    UserAnswer,
    UserScenarioScore,
)
from authoringtool.tasks import compute_student_performance_metrics
from usergroups.models import UserGroup, UserGroupMembership


# (url name, task attribute imported into authoringtool.views)
TASK_ENDPOINTS = [
    ('sankey_data', 'compute_sankey_data'),
    ('final_performance', 'compute_final_performance'),
    ('activity_answer_data', 'compute_activity_answers_data'),
    ('performance_data', 'compute_performance_data'),
    ('time_spent_data', 'compute_time_spent_data'),
    ('detailed_phase_scores_data', 'compute_detailed_phase_scores_data'),
    ('performers_data', 'compute_performers_data'),
    ('time_spent_by_performer_type', 'compute_time_spent_by_performer_type'),
    ('scenario_paths', 'compute_scenario_paths'),
    ('student_performance_metrics', 'compute_student_performance_metrics'),
]

SYNC_ENDPOINTS = ['lab_sessions_data', 'performance_by_department']


def mocked_task(task_id=None):
    task = MagicMock()
    task.delay.return_value.id = task_id or str(uuid.uuid4())
    return task


class AnalyticsAccessTests(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('analytics_owner', password='pass')
        self.other_teacher = User.objects.create_user('analytics_other', password='pass')
        self.student = User.objects.create_user('analytics_student', password='pass')
        for teacher in (self.owner, self.other_teacher):
            teacher.groups.add(teachers)

        self.public_scenario = Scenario.objects.create(
            name='Public analytics', visibility_status='public',
            created_by=self.owner, updated_by=self.owner,
        )
        self.private_scenario = Scenario.objects.create(
            name='Private analytics', visibility_status='private',
            created_by=self.owner, updated_by=self.owner,
        )
        self.owner_group = UserGroup.objects.create(
            name='Owner class', prefix='own', number_of_users=0,
            created_by=self.owner,
        )
        self.other_group = UserGroup.objects.create(
            name='Other class', prefix='oth', number_of_users=0,
            created_by=self.other_teacher,
        )

    def get(self, name, scenario, **params):
        return self.client.get(reverse(name, args=[scenario.id]), params)

    def test_students_cannot_start_any_analytics_task(self):
        self.client.force_login(self.student)
        for name, task_name in TASK_ENDPOINTS:
            with self.subTest(endpoint=name), \
                    patch(f'authoringtool.views.{task_name}', mocked_task()) as task:
                response = self.get(name, self.public_scenario)
                self.assertEqual(response.status_code, 403)
                task.delay.assert_not_called()

    def test_students_cannot_read_synchronous_analytics(self):
        self.client.force_login(self.student)
        for name in SYNC_ENDPOINTS:
            with self.subTest(endpoint=name):
                self.assertEqual(self.get(name, self.public_scenario).status_code, 403)

    def test_teacher_cannot_analyse_private_scenario_of_someone_else(self):
        self.client.force_login(self.other_teacher)
        for name, task_name in TASK_ENDPOINTS:
            with self.subTest(endpoint=name), \
                    patch(f'authoringtool.views.{task_name}', mocked_task()) as task:
                response = self.get(name, self.private_scenario)
                self.assertEqual(response.status_code, 403)
                task.delay.assert_not_called()
        for name in SYNC_ENDPOINTS:
            with self.subTest(endpoint=name):
                self.assertEqual(self.get(name, self.private_scenario).status_code, 403)

    def test_non_owner_cannot_filter_by_another_teachers_group(self):
        self.client.force_login(self.other_teacher)
        with patch('authoringtool.views.compute_performance_data', mocked_task()) as task:
            response = self.get(
                'performance_data', self.public_scenario,
                group_ids=str(self.owner_group.id),
            )
        self.assertEqual(response.status_code, 403)
        task.delay.assert_not_called()

    def test_non_owner_cannot_filter_help_time_by_another_teachers_group(self):
        self.client.force_login(self.other_teacher)
        with patch('authoringtool.views.help_time_chart', return_value={}) as chart:
            response = self.get(
                'help_time_data', self.public_scenario,
                group_ids=str(self.owner_group.id),
            )
        self.assertEqual(response.status_code, 403)
        chart.assert_not_called()

    def test_non_owner_can_filter_by_own_group(self):
        self.client.force_login(self.other_teacher)
        with patch('authoringtool.views.compute_performance_data', mocked_task()) as task:
            response = self.get(
                'performance_data', self.public_scenario,
                group_ids=str(self.other_group.id),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(task.delay.call_args.args[1], [self.other_group.id])

    def test_non_owner_aggregates_without_group_filter_cover_all_students(self):
        self.client.force_login(self.other_teacher)
        with patch('authoringtool.views.compute_performance_data', mocked_task()) as task:
            response = self.get('performance_data', self.public_scenario)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(task.delay.call_args.args[1], [])

    def per_student_call(self, user):
        self.client.force_login(user)
        with patch(
            'authoringtool.views.compute_student_performance_metrics',
            mocked_task(),
        ) as task:
            response = self.get('student_performance_metrics', self.public_scenario)
        self.assertEqual(response.status_code, 200)
        # (scenario_id, group_ids, start, end, include_detail, named_user_ids)
        return task.delay.call_args.args

    def test_non_owner_per_student_metrics_name_only_own_students(self):
        own_student = User.objects.create_user('analytics_own_student')
        UserGroupMembership.objects.create(group=self.other_group, user=own_student)
        args = self.per_student_call(self.other_teacher)
        self.assertEqual(args[1], [])
        self.assertEqual(args[5], [own_student.id])

    def test_non_owner_without_groups_sees_every_student_pseudonymised(self):
        self.other_group.delete()
        args = self.per_student_call(self.other_teacher)
        self.assertEqual(args[1], [])
        self.assertEqual(args[5], [])

    def test_owner_sees_every_student_by_name(self):
        args = self.per_student_call(self.owner)
        self.assertEqual(args[1], [])
        self.assertIsNone(args[5])

    def test_owner_may_filter_by_any_group(self):
        self.client.force_login(self.owner)
        with patch('authoringtool.views.compute_sankey_data', mocked_task()) as task:
            response = self.get(
                'sankey_data', self.public_scenario,
                group_ids=f'{self.other_group.id},{self.owner_group.id}',
            )
        self.assertEqual(response.status_code, 200)
        # The sankey task still receives its group filter as a comma string.
        self.assertEqual(
            task.delay.call_args.args[1],
            f'{self.other_group.id},{self.owner_group.id}',
        )


class AnalyticsTaskStatusTests(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('status_owner', password='pass')
        self.other_teacher = User.objects.create_user('status_other', password='pass')
        for teacher in (self.owner, self.other_teacher):
            teacher.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='Status analytics', visibility_status='public',
            created_by=self.owner, updated_by=self.owner,
        )
        self.task_id = str(uuid.uuid4())
        self.client.force_login(self.owner)
        with patch(
            'authoringtool.views.compute_student_performance_metrics',
            mocked_task(self.task_id),
        ):
            response = self.client.get(
                reverse('student_performance_metrics', args=[self.scenario.id])
            )
        self.assertEqual(response.status_code, 200)

    def poll(self):
        finished = MagicMock(state='SUCCESS', result={'rows': ['secret']})
        with patch('authoringtool.views.AsyncResult', return_value=finished):
            return self.client.get(reverse(
                'get_student_performance_metrics_task_status',
                args=[self.task_id],
            ))

    def test_task_starter_can_read_result(self):
        response = self.poll()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['data'], {'rows': ['secret']})

    def test_other_teacher_cannot_read_result(self):
        self.client.force_login(self.other_teacher)
        self.assertEqual(self.poll().status_code, 404)


class StudentMetricsPseudonymisationTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_user('pseudo_admin')
        department = SchoolDepartment.objects.create(name='Pseudo Dept')
        self.scenario = Scenario.objects.create(
            name='Pseudonymised metrics', created_by=self.admin, updated_by=self.admin,
        )
        phase = Phase.objects.create(
            name='Phase 1', scenario=self.scenario,
            created_by=self.admin, updated_by=self.admin,
        )
        quiz_type = ActivityType.objects.create(
            name='Quiz', created_by=self.admin, updated_by=self.admin,
        )
        quiz = Activity.objects.create(
            name='Quiz', text='Question', is_evaluatable=True, is_primary_ev=True,
            phase=phase, scenario=self.scenario, activity_type=quiz_type,
            created_by=self.admin, updated_by=self.admin,
        )
        correct = Answer.objects.create(
            activity=quiz, text='Right', is_correct=True, answer_weight=10,
            created_by=self.admin, updated_by=self.admin,
        )
        self.own_student = User.objects.create_user('pseudo_own_student')
        self.other_student = User.objects.create_user('pseudo_other_school_student')
        for student in (self.own_student, self.other_student):
            student.school_department = department
            student.save()
            UserScenarioScore.objects.create(
                user=student, scenario=self.scenario, user_score=10,
            )
            UserAnswer.objects.create(
                user=student, activity=quiz, answer=correct, timing=30,
            )

    def rows(self, named_user_ids):
        result = compute_student_performance_metrics(
            self.scenario.id, [], None, None, False, named_user_ids,
        )
        return result['csv_content'], list(csv.DictReader(io.StringIO(result['csv_content'])))

    def test_students_outside_the_viewers_groups_are_pseudonymised(self):
        content, rows = self.rows([self.own_student.id])
        by_name = {row['Username']: row for row in rows}
        self.assertEqual(by_name[self.own_student.username]['User ID'], str(self.own_student.id))
        self.assertIn('Student 1', by_name)
        self.assertEqual(by_name['Student 1']['User ID'], '')
        self.assertNotIn(self.other_student.username, content)
        self.assertEqual(len(rows), 2)

    def test_pseudonymised_rows_show_only_the_start_date(self):
        _, rows = self.rows([self.own_student.id])
        by_name = {row['Username']: row for row in rows}
        named_start = by_name[self.own_student.username]['Phase 1 Start Time']
        hidden_start = by_name['Student 1']['Phase 1 Start Time']
        self.assertRegex(named_start, r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')
        self.assertRegex(hidden_start, r'^\d{4}-\d{2}-\d{2}$')

    def test_everyone_is_named_when_no_restriction_is_given(self):
        content, rows = self.rows(None)
        self.assertEqual(
            {row['Username'] for row in rows},
            {self.own_student.username, self.other_student.username},
        )
