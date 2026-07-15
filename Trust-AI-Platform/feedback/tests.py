import json

from django.contrib.auth.models import Group, User
from django.db import IntegrityError, transaction
from django.test import Client, TestCase
from django.urls import reverse

from authoringtool.models import Scenario
from feedback.models import FeedbackAnswer, FeedbackForm, FeedbackQuestion, FeedbackResponse


class FeedbackFormAppliesToTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('fb_admin', password='pass', is_staff=True)
        self.scenario_a = Scenario.objects.create(name='FB Scenario A', created_by=self.user, updated_by=self.user)
        self.scenario_b = Scenario.objects.create(name='FB Scenario B', created_by=self.user, updated_by=self.user)

    def test_assign_to_all_applies_everywhere(self):
        form = FeedbackForm.objects.create(title='All', audience='student', assign_to_all=True, created_by=self.user)
        self.assertTrue(form.applies_to(self.scenario_a))
        self.assertTrue(form.applies_to(self.scenario_b))

    def test_assign_to_all_respects_exclusions(self):
        form = FeedbackForm.objects.create(title='All minus B', audience='student', assign_to_all=True, created_by=self.user)
        form.excluded_scenarios.add(self.scenario_b)
        self.assertTrue(form.applies_to(self.scenario_a))
        self.assertFalse(form.applies_to(self.scenario_b))

    def test_explicit_inclusion_mode(self):
        form = FeedbackForm.objects.create(title='Only A', audience='student', assign_to_all=False, created_by=self.user)
        form.included_scenarios.add(self.scenario_a)
        self.assertTrue(form.applies_to(self.scenario_a))
        self.assertFalse(form.applies_to(self.scenario_b))

    def test_inactive_form_never_applies(self):
        form = FeedbackForm.objects.create(title='Off', audience='student', assign_to_all=True, is_active=False, created_by=self.user)
        self.assertFalse(form.applies_to(self.scenario_a))


class FeedbackResponseConstraintTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('fb_responder', password='pass')
        self.scenario = Scenario.objects.create(name='FB Constraint Scenario', created_by=self.user, updated_by=self.user)
        self.form = FeedbackForm.objects.create(title='F', audience='student', created_by=self.user)

    def test_one_response_per_form_user_scenario(self):
        FeedbackResponse.objects.create(form=self.form, user=self.user, scenario=self.scenario)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                FeedbackResponse.objects.create(form=self.form, user=self.user, scenario=self.scenario)

    def test_one_answer_per_question_per_response(self):
        question = FeedbackQuestion.objects.create(form=self.form, text='Q1', question_type='text')
        response = FeedbackResponse.objects.create(form=self.form, user=self.user, scenario=self.scenario)
        FeedbackAnswer.objects.create(response=response, question=question, answer_text='a')
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                FeedbackAnswer.objects.create(response=response, question=question, answer_text='b')

    def test_questions_ordered_by_order_field(self):
        FeedbackQuestion.objects.create(form=self.form, text='Second', question_type='text', order=2)
        FeedbackQuestion.objects.create(form=self.form, text='First', question_type='text', order=1)
        texts = list(self.form.questions.values_list('text', flat=True))
        self.assertEqual(texts, ['First', 'Second'])


class FeedbackSubmitEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.teacher = User.objects.create_user('fb_teacher', password='pass')
        self.teacher.groups.add(teachers)
        self.student = User.objects.create_user('fb_student', password='pass')
        self.scenario = Scenario.objects.create(name='FB Submit Scenario', created_by=self.teacher, updated_by=self.teacher)

        self.student_form = FeedbackForm.objects.create(title='Student form', audience='student', created_by=self.teacher)
        self.q_choice = FeedbackQuestion.objects.create(
            form=self.student_form, text='Useful?', question_type='choice',
            options=['Yes', 'No'], is_required=True, order=1,
        )
        self.q_text = FeedbackQuestion.objects.create(
            form=self.student_form, text='Comments?', question_type='text',
            is_required=False, order=2,
        )
        self.teacher_form = FeedbackForm.objects.create(title='Teacher form', audience='teacher', created_by=self.teacher)
        self.tq = FeedbackQuestion.objects.create(
            form=self.teacher_form, text='Proposals good?', question_type='choice',
            options=['Yes', 'No'], is_required=True, order=1,
        )

    def _submit(self, form, answers):
        url = reverse('feedback_submit', args=[form.id, self.scenario.id])
        return self.client.post(url, json.dumps({'answers': answers}), content_type='application/json')

    def test_student_can_submit_student_form(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit(self.student_form, {str(self.q_choice.id): 'Yes', str(self.q_text.id): 'Great tool'})
        self.assertTrue(r.json()['success'])
        response = FeedbackResponse.objects.get(form=self.student_form, user=self.student, scenario=self.scenario)
        self.assertEqual(response.answers.count(), 2)

    def test_teacher_blocked_from_student_form(self):
        self.client.login(username='fb_teacher', password='pass')
        r = self._submit(self.student_form, {str(self.q_choice.id): 'Yes'})
        self.assertEqual(r.status_code, 403)
        self.assertFalse(FeedbackResponse.objects.filter(form=self.student_form).exists())

    def test_student_blocked_from_teacher_form(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit(self.teacher_form, {str(self.tq.id): 'Yes'})
        self.assertEqual(r.status_code, 403)

    def test_form_not_applicable_to_scenario_blocked(self):
        self.student_form.assign_to_all = False
        self.student_form.save()
        self.client.login(username='fb_student', password='pass')
        r = self._submit(self.student_form, {str(self.q_choice.id): 'Yes'})
        self.assertEqual(r.status_code, 403)

    def test_missing_required_answer_rejected(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit(self.student_form, {str(self.q_text.id): 'only optional'})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(FeedbackResponse.objects.filter(form=self.student_form).exists())

    def test_choice_answer_must_be_valid_option(self):
        self.client.login(username='fb_student', password='pass')
        r = self._submit(self.student_form, {str(self.q_choice.id): 'Maybe'})
        self.assertEqual(r.status_code, 400)

    def test_duplicate_submission_returns_friendly_error(self):
        self.client.login(username='fb_student', password='pass')
        self._submit(self.student_form, {str(self.q_choice.id): 'Yes'})
        r = self._submit(self.student_form, {str(self.q_choice.id): 'No'})
        self.assertEqual(r.status_code, 400)
        self.assertIn('already', r.json()['error'].lower())
        self.assertEqual(FeedbackResponse.objects.filter(form=self.student_form).count(), 1)

    def test_get_method_not_allowed(self):
        self.client.login(username='fb_student', password='pass')
        url = reverse('feedback_submit', args=[self.student_form.id, self.scenario.id])
        r = self.client.get(url)
        self.assertEqual(r.status_code, 405)


class FeedbackUtilsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('fb_utils', password='pass')
        self.scenario = Scenario.objects.create(name='FB Utils Scenario', created_by=self.user, updated_by=self.user)

    def test_get_applicable_form_returns_newest_applicable(self):
        from feedback.utils import get_applicable_form
        older = FeedbackForm.objects.create(title='Older', audience='student', created_by=self.user)
        newer = FeedbackForm.objects.create(title='Newer', audience='student', created_by=self.user)
        self.assertEqual(get_applicable_form(self.scenario, 'student'), newer)

    def test_get_applicable_form_skips_wrong_audience_and_inactive(self):
        from feedback.utils import get_applicable_form
        FeedbackForm.objects.create(title='Teacher only', audience='teacher', created_by=self.user)
        FeedbackForm.objects.create(title='Inactive', audience='student', is_active=False, created_by=self.user)
        self.assertIsNone(get_applicable_form(self.scenario, 'student'))

    def test_serialize_form_shape(self):
        from feedback.utils import serialize_form
        form = FeedbackForm.objects.create(title='S', description='D', audience='student', created_by=self.user)
        FeedbackQuestion.objects.create(form=form, text='Q', question_type='choice', options=['A', 'B'], order=1)
        data = serialize_form(form)
        self.assertEqual(data['title'], 'S')
        self.assertEqual(len(data['questions']), 1)
        self.assertEqual(data['questions'][0]['options'], ['A', 'B'])
        self.assertEqual(data['questions'][0]['type'], 'choice')
