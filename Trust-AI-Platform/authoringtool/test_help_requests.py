import csv
import io
from datetime import timedelta

from django.contrib.auth.models import Group, User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from authoringtool.help_time import help_totals
from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    HelpRequest,
    NextQuestionLogic,
    Phase,
    Scenario,
    ScenarioImplementation,
    SchoolDepartment,
    UserAnswer,
    UserScenarioScore,
)
from authoringtool.tasks import (
    compute_student_performance_metrics,
    compute_time_spent_data,
)
from usergroups.models import UserGroup


class HelpRequestBase(TestCase):
    def setUp(self):
        self.teachers = Group.objects.create(name='teachers')
        self.teacher = User.objects.create_user('help_teacher', password='pass')
        self.teacher.groups.add(self.teachers)
        self.department = SchoolDepartment.objects.create(name='Physics')
        self.student = self.make_student('help_student')
        self.question_type = ActivityType.objects.create(
            name='Question',
            created_by=self.teacher,
            updated_by=self.teacher,
        )
        self.scenario = Scenario.objects.create(
            name='Pendulum help',
            language='English',
            visibility_status='public',
            created_by=self.teacher,
            updated_by=self.teacher,
        )
        self.explore = Phase.objects.create(
            name='Explore', scenario=self.scenario,
            created_by=self.teacher, updated_by=self.teacher,
        )
        self.explain = Phase.objects.create(
            name='Explain', scenario=self.scenario,
            created_by=self.teacher, updated_by=self.teacher,
        )
        self.first = self.make_question(self.explore, 'What is the period?')
        self.second = self.make_question(self.explain, 'Why does mass not matter?')
        for answer in self.first.answers.all():
            NextQuestionLogic.objects.create(
                activity=self.first, answer=answer, next_activity=self.second,
            )
        self.student_group().assigned_scenarios.add(self.scenario)
        self.scenario.refresh_from_db()

    def make_student(self, username):
        student = User.objects.create_user(username, password='pass')
        student.school_department = self.department
        student.save()
        self.student_group().members.add(student)
        return student

    def student_group(self):
        # Students only reach scenarios assigned to their group; the
        # scenario is assigned once it exists (see setUp).
        group, _ = UserGroup.objects.get_or_create(
            name='Help class',
            defaults={'prefix': 'hlp', 'number_of_users': 0, 'created_by': self.teacher},
        )
        return group

    def make_question(self, phase, text):
        activity = Activity.objects.create(
            name=text, text=text, plain_text=text,
            scenario=self.scenario, phase=phase,
            activity_type=self.question_type, is_evaluatable=False,
            created_by=self.teacher, updated_by=self.teacher,
        )
        Answer.objects.create(
            activity=activity, text='A', is_correct=True,
            created_by=self.teacher, updated_by=self.teacher,
        )
        Answer.objects.create(
            activity=activity, text='B', is_correct=False,
            created_by=self.teacher, updated_by=self.teacher,
        )
        return activity

    def answer(self, student, activity, timing):
        implementation, _ = ScenarioImplementation.start_or_resume(
            student, self.scenario,
        )
        if not UserScenarioScore.objects.filter(
            implementation=implementation,
        ).exists():
            UserScenarioScore.objects.create(
                user=student, scenario=self.scenario,
                implementation=implementation,
            )
        return UserAnswer.objects.create(
            user=student, activity=activity,
            answer=activity.answers.order_by('id').first(),
            implementation=implementation, timing=timing,
        )

    def help_period(self, student, activity, seconds):
        ended = timezone.now()
        return HelpRequest.objects.create(
            user=student, scenario=self.scenario,
            activity=activity, phase=activity.phase,
            started_at=ended - timedelta(seconds=seconds),
            ended_at=ended, duration_seconds=seconds,
        )


class StudentPauseTests(HelpRequestBase):
    def start(self, user, activity):
        self.client.force_login(user)
        return self.client.post(
            reverse('help_start', args=[self.scenario.id]),
            {'activity_id': activity.id},
        )

    def test_student_pause_opens_help_period_for_current_activity(self):
        response = self.start(self.student, self.first)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['paused'])
        request = HelpRequest.objects.get()
        self.assertEqual(request.user, self.student)
        self.assertEqual(request.activity, self.first)
        self.assertEqual(request.phase, self.explore)
        self.assertIsNone(request.ended_at)

    def test_second_pause_does_not_open_another_period(self):
        self.start(self.student, self.first)
        self.start(self.student, self.first)

        self.assertEqual(HelpRequest.objects.count(), 1)

    def test_resume_closes_period_and_records_seconds(self):
        self.start(self.student, self.first)
        HelpRequest.objects.update(
            started_at=timezone.now() - timedelta(seconds=45),
        )

        response = self.client.post(
            reverse('help_end', args=[self.scenario.id]),
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['paused'])
        request = HelpRequest.objects.get()
        self.assertIsNotNone(request.ended_at)
        self.assertGreaterEqual(request.duration_seconds, 45)
        self.assertLess(request.duration_seconds, 60)

    def test_resume_without_pause_is_harmless(self):
        self.client.force_login(self.student)

        response = self.client.post(
            reverse('help_end', args=[self.scenario.id]),
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(HelpRequest.objects.exists())

    def test_teacher_cannot_pause(self):
        response = self.start(self.teacher, self.first)

        self.assertEqual(response.status_code, 403)
        self.assertFalse(HelpRequest.objects.exists())

    def test_activity_from_another_scenario_is_rejected(self):
        other = Scenario.objects.create(
            name='Other', created_by=self.teacher, updated_by=self.teacher,
        )
        foreign = Activity.objects.create(
            name='Foreign', text='Foreign', scenario=other,
            created_by=self.teacher, updated_by=self.teacher,
        )

        response = self.start(self.student, foreign)

        self.assertEqual(response.status_code, 400)
        self.assertFalse(HelpRequest.objects.exists())

    def test_pause_requires_post(self):
        self.client.force_login(self.student)

        response = self.client.get(
            reverse('help_start', args=[self.scenario.id]),
        )

        self.assertEqual(response.status_code, 405)


class StudentPageTests(HelpRequestBase):
    def test_student_sees_ask_for_help_button(self):
        self.client.force_login(self.student)

        response = self.client.get(
            reverse('studentView', args=[self.scenario.id]),
        )

        self.assertContains(response, 'Ask for help')
        self.assertFalse(response.context['help_paused'])

    def test_help_texts_follow_scenario_language(self):
        self.scenario.language = 'Greek'
        self.scenario.save(update_fields=['language'])
        self.client.force_login(self.student)

        response = self.client.get(
            reverse('studentView', args=[self.scenario.id]),
        )

        self.assertContains(response, 'Ζήτησε βοήθεια')
        self.assertContains(response, 'Παύση: ζητήθηκε βοήθεια')
        self.assertNotContains(response, 'Paused: help requested')

    def test_unknown_language_falls_back_to_english(self):
        self.scenario.language = 'Klingon'
        self.scenario.save(update_fields=['language'])
        self.client.force_login(self.student)

        response = self.client.get(
            reverse('studentView', args=[self.scenario.id]),
        )

        self.assertContains(response, 'Ask for help')

    def test_teacher_does_not_see_ask_for_help_button(self):
        self.client.force_login(self.teacher)

        response = self.client.get(
            reverse('studentView', args=[self.scenario.id]),
        )

        self.assertNotContains(response, 'Ask for help')

    def test_open_pause_is_restored_after_refresh(self):
        HelpRequest.objects.create(
            user=self.student, scenario=self.scenario,
            activity=self.first, phase=self.explore,
        )
        self.client.force_login(self.student)

        response = self.client.get(
            reverse('studentView', args=[self.scenario.id]),
        )

        self.assertTrue(response.context['help_paused'])


class HelpTotalsTests(HelpRequestBase):
    def test_totals_per_activity_phase_and_scenario(self):
        other = self.make_student('help_other')
        self.help_period(self.student, self.first, 30)
        self.help_period(self.student, self.first, 10)
        self.help_period(other, self.second, 20)

        totals = help_totals(self.scenario)

        self.assertEqual(totals['by_activity'][self.first.id]['seconds'], 40)
        self.assertEqual(totals['by_activity'][self.first.id]['requests'], 2)
        self.assertEqual(totals['by_activity'][self.first.id]['students'], 1)
        self.assertEqual(totals['by_phase'][self.explain.id]['seconds'], 20)
        self.assertEqual(totals['scenario']['seconds'], 60)
        self.assertEqual(totals['scenario']['students'], 2)
        self.assertEqual(
            totals['by_user_activity'][(self.student.id, self.first.id)],
            40,
        )

    def test_open_periods_are_not_counted(self):
        HelpRequest.objects.create(
            user=self.student, scenario=self.scenario,
            activity=self.first, phase=self.explore,
        )

        totals = help_totals(self.scenario)

        self.assertEqual(totals['scenario']['seconds'], 0)

    def test_user_filter_limits_totals(self):
        other = self.make_student('help_other')
        self.help_period(self.student, self.first, 30)
        self.help_period(other, self.first, 50)

        totals = help_totals(self.scenario, user_ids=[other.id])

        self.assertEqual(totals['scenario']['seconds'], 50)


class ActiveTimeTests(HelpRequestBase):
    def test_average_time_spent_excludes_help_time(self):
        other = self.make_student('help_other')
        self.answer(self.student, self.first, 100)
        self.answer(other, self.first, 100)
        self.help_period(self.student, self.first, 30)

        data = compute_time_spent_data(
            self.scenario.id, [], None, None, 'activities_timing',
        )

        index = data['categories'].index(self.first.name)
        self.assertEqual(data['time_spent'][index], 85)


class HelpCsvTests(HelpRequestBase):
    def rows(self):
        result = compute_student_performance_metrics(
            self.scenario.id, [], None, None, include_activity_detail=True,
        )
        return list(csv.DictReader(io.StringIO(result['csv_content'])))

    def test_csv_has_help_columns_and_active_times(self):
        self.answer(self.student, self.first, 100)
        self.answer(self.student, self.second, 60)
        self.help_period(self.student, self.first, 30)

        row = self.rows()[0]

        self.assertEqual(row['Explore Time'], '70')
        self.assertEqual(row['Explore Help Time (s)'], '30')
        self.assertEqual(row['Explain Help Time (s)'], '0')
        self.assertEqual(row['Total Help Time (s)'], '30')
        self.assertEqual(row['Help Requests'], '1')
        prefix = f'Explore > {self.first.name}'
        self.assertEqual(row[f'{prefix} Time (s)'], '70')
        self.assertEqual(row[f'{prefix} Help Time (s)'], '30')


class HelpDashboardEndpointTests(HelpRequestBase):
    def test_teacher_gets_average_help_time_per_activity(self):
        other = self.make_student('help_other')
        self.answer(self.student, self.first, 100)
        self.answer(other, self.first, 100)
        self.help_period(self.student, self.first, 30)
        self.client.force_login(self.teacher)

        response = self.client.get(
            reverse('help_time_data', args=[self.scenario.id]),
            {'level': 'activity'},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        index = data['categories'].index(self.first.name)
        self.assertEqual(data['average_help_seconds'][index], 15)
        self.assertEqual(data['requests'][index], 1)
        self.assertEqual(data['students_asking'][index], 1)

    def test_scenario_level_returns_one_total(self):
        self.answer(self.student, self.first, 100)
        self.help_period(self.student, self.first, 30)
        self.help_period(self.student, self.second, 10)
        self.client.force_login(self.teacher)

        data = self.client.get(
            reverse('help_time_data', args=[self.scenario.id]),
            {'level': 'scenario'},
        ).json()

        self.assertEqual(data['categories'], [self.scenario.name])
        self.assertEqual(data['average_help_seconds'], [40])

    def test_dashboard_shows_help_time_card(self):
        self.client.force_login(self.teacher)

        response = self.client.get(reverse('index'))

        self.assertContains(response, 'Help Time (Ask for help)')
        self.assertContains(response, '/authoringtool/help_time_data/')

    def test_students_cannot_read_help_analytics(self):
        self.client.force_login(self.student)

        response = self.client.get(
            reverse('help_time_data', args=[self.scenario.id]),
        )

        self.assertNotEqual(response.status_code, 200)
