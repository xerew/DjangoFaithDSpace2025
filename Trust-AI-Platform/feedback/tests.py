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
