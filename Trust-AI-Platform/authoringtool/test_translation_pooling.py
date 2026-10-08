import csv
import tempfile

from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityType,
    Answer,
    NextQuestionLogic,
    Phase,
    Scenario,
    ScenarioImplementation,
    SchoolDepartment,
    TranslationMatch,
    UserAnswer,
    UserScenarioScore,
)
from authoringtool.tasks import compute_category_metrics_per_phase_activity
from authoringtool.utils import get_scenario_evidence_cache_paths


class PoolingBase(TestCase):
    def setUp(self):
        self.teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('pool_owner', password='pass')
        self.owner.groups.add(self.teachers)
        self.department = SchoolDepartment.objects.create(name='Physics')
        self.question_type = ActivityType.objects.create(
            name='Question', created_by=self.owner, updated_by=self.owner,
        )
        self.explanation_type = ActivityType.objects.create(
            name='Explanation', created_by=self.owner, updated_by=self.owner,
        )
        self.student_number = 0

    def build(self, name, language, visibility='public', owner=None):
        owner = owner or self.owner
        scenario = Scenario.objects.create(
            name=name, language=language, visibility_status=visibility,
            created_by=owner, updated_by=owner,
        )
        phase = Phase.objects.create(
            name='Phase', scenario=scenario,
            created_by=owner, updated_by=owner,
        )
        question = Activity.objects.create(
            name=f'{name} question', text='Period?', plain_text='Period?',
            scenario=scenario, phase=phase, activity_type=self.question_type,
            is_evaluatable=True, is_primary_ev=True,
            created_by=owner, updated_by=owner,
        )
        Answer.objects.create(
            activity=question, text='Right', is_correct=True, answer_weight=1,
        )
        Answer.objects.create(
            activity=question, text='Wrong', is_correct=False, answer_weight=0,
        )
        explanation = Activity.objects.create(
            name=f'{name} explanation', text='Because', plain_text='Because',
            scenario=scenario, phase=phase,
            activity_type=self.explanation_type,
            created_by=owner, updated_by=owner,
        )
        for answer in question.answers.all():
            NextQuestionLogic.objects.create(
                activity=question, answer=answer, next_activity=explanation,
            )
        scenario.refresh_from_db()
        return scenario, question, explanation

    def confirm(self, first, second):
        (scenario_a, question_a, explanation_a), (scenario_b, question_b, explanation_b) = sorted(
            (first, second), key=lambda built: built[0].id,
        )
        return TranslationMatch.objects.create(
            scenario_a=scenario_a, scenario_b=scenario_b,
            confidence=100, exact_match=True, status='confirmed',
            activity_map=[
                [question_a.id, question_b.id],
                [explanation_a.id, explanation_b.id],
            ],
        )

    def student_answers(self, built, correct, timing):
        scenario, question, _ = built
        self.student_number += 1
        student = User.objects.create_user(f'pool_student_{self.student_number}')
        student.school_department = self.department
        student.save()
        implementation, _ = ScenarioImplementation.start_or_resume(student, scenario)
        UserScenarioScore.objects.create(
            user=student, scenario=scenario, implementation=implementation,
        )
        UserAnswer.objects.create(
            user=student, activity=question,
            answer=question.answers.get(is_correct=correct),
            implementation=implementation, timing=timing,
        )
        return student


@override_settings(AI_METRICS_CACHE_ROOT=tempfile.mkdtemp())
class PooledMetricsTests(PoolingBase):
    def question_rows(self, built):
        scenario, question, _ = built
        compute_category_metrics_per_phase_activity(scenario.id)
        path = get_scenario_evidence_cache_paths(scenario, 'local')['metrics']
        with open(path, encoding='utf-8') as handle:
            return [
                row for row in csv.DictReader(handle)
                if row['Activity'] == question.name and row['Total'] not in ('', '0')
            ]

    def test_confirmed_translation_answers_are_combined(self):
        english = self.build('Pendulum', 'English')
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        self.confirm(english, greek)
        self.student_answers(english, correct=True, timing=10)
        self.student_answers(greek, correct=False, timing=100)
        self.student_answers(greek, correct=False, timing=100)

        rows = self.question_rows(english)

        self.assertEqual(sum(int(row['Total']) for row in rows), 3)
        self.assertEqual(sum(int(row['Correct'] or 0) for row in rows), 1)
        self.assertEqual(sum(int(row['Wrong'] or 0) for row in rows), 2)

    def test_time_uses_same_language_students_only(self):
        english = self.build('Pendulum', 'English')
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        self.confirm(english, greek)
        self.student_answers(english, correct=True, timing=10)
        self.student_answers(greek, correct=True, timing=100)

        rows = self.question_rows(english)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['Total'], '2')
        self.assertEqual(float(rows[0]['Avg Time']), 10.0)

    def test_translations_of_translations_are_combined(self):
        english = self.build('Pendulum', 'English')
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        romanian = self.build('Pendulul', 'Romanian')
        self.confirm(english, greek)
        self.confirm(greek, romanian)
        self.student_answers(english, correct=True, timing=10)
        self.student_answers(romanian, correct=False, timing=50)

        rows = self.question_rows(english)

        self.assertEqual(sum(int(row['Total']) for row in rows), 2)

    def test_unconfirmed_translation_is_not_combined(self):
        english = self.build('Pendulum', 'English')
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        self.confirm(english, greek)
        TranslationMatch.objects.update(status='pending')
        self.student_answers(english, correct=True, timing=10)
        self.student_answers(greek, correct=False, timing=100)

        rows = self.question_rows(english)

        self.assertEqual(sum(int(row['Total']) for row in rows), 1)

    def test_new_translation_answers_change_the_cache_file(self):
        english = self.build('Pendulum', 'English')
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        self.confirm(english, greek)
        self.student_answers(english, correct=True, timing=10)
        before = get_scenario_evidence_cache_paths(english[0], 'local')['metrics']

        self.student_answers(greek, correct=False, timing=100)

        after = get_scenario_evidence_cache_paths(english[0], 'local')['metrics']
        self.assertNotEqual(before, after)


class CombinedThresholdTests(PoolingBase):
    def setUp(self):
        super().setUp()
        self.english = self.build('Pendulum', 'English')
        self.greek = self.build('Εκκρεμές', 'Ελληνικά')
        Scenario.objects.filter(pk=self.english[0].pk).update(
            ai_metrics_min_implementations=3,
        )
        self.english[0].refresh_from_db()
        self.student_answers(self.english, correct=True, timing=10)
        self.student_answers(self.greek, correct=True, timing=10)
        self.student_answers(self.greek, correct=False, timing=10)
        self.client.force_login(self.owner)

    def test_scenario_page_uses_combined_count_for_metrics_button(self):
        self.confirm(self.english, self.greek)

        response = self.client.get(
            reverse('viewScenario', args=[self.english[0].id]),
        )

        self.assertEqual(response.context['implementation_count'], 1)
        self.assertEqual(response.context['proposal_threshold_count'], 3)
        self.assertNotContains(response, 'data-bs-target="#lowDataWarningModal"')

    def test_translations_modal_is_available_above_the_threshold(self):
        self.confirm(self.english, self.greek)

        response = self.client.get(
            reverse('viewScenario', args=[self.english[0].id]),
        )

        self.assertContains(response, 'id="translationsModal"')
        self.assertContains(response, 'Translations (1)')

    def test_without_translations_own_count_is_used(self):
        response = self.client.get(
            reverse('viewScenario', args=[self.english[0].id]),
        )

        self.assertEqual(response.context['proposal_threshold_count'], 1)
        self.assertContains(response, 'data-bs-target="#lowDataWarningModal"')

    def test_metrics_page_uses_combined_count_for_proposals(self):
        self.confirm(self.english, self.greek)

        response = self.client.get(
            reverse('ai_metrics', kwargs={'scenario_id': self.english[0].id}),
        )

        self.assertEqual(response.context['proposal_implementation_count'], 3)
        self.assertEqual(response.context['implementation_count'], 1)


class ScenarioListTests(PoolingBase):
    def setUp(self):
        super().setUp()
        self.client.force_login(self.owner)

    def listed(self, **params):
        response = self.client.get(reverse('scenarios'), params)
        return response, [scenario.id for scenario in response.context['myScenarios']]

    def test_only_english_version_is_listed_with_translations_button(self):
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        english = self.build('Pendulum', 'English')
        self.confirm(english, greek)

        response, ids = self.listed()

        self.assertIn(english[0].id, ids)
        self.assertNotIn(greek[0].id, ids)
        self.assertContains(response, 'Translations (1)')
        self.assertContains(response, 'Εκκρεμές')

    def test_group_without_english_shows_the_original(self):
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        romanian = self.build('Pendulul', 'Romanian')
        self.confirm(greek, romanian)

        _, ids = self.listed()

        self.assertIn(greek[0].id, ids)
        self.assertNotIn(romanian[0].id, ids)

    def test_language_filter_shows_the_matching_version(self):
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        english = self.build('Pendulum', 'English')
        self.confirm(english, greek)

        _, ids = self.listed(language='Ελληνικά')

        self.assertEqual(ids, [greek[0].id])

    def test_unconfirmed_translations_are_listed_separately(self):
        greek = self.build('Εκκρεμές', 'Ελληνικά')
        english = self.build('Pendulum', 'English')
        self.confirm(english, greek)
        TranslationMatch.objects.update(status='rejected')

        _, ids = self.listed()

        self.assertIn(english[0].id, ids)
        self.assertIn(greek[0].id, ids)
