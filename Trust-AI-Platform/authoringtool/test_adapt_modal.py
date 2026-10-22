"""Adapt: the confirmation modal, the chosen name, and an all-or-nothing copy."""

from unittest.mock import patch

from django.contrib.auth.models import Group, User
from django.test import TestCase
from django.urls import reverse

from .models import Activity, ActivityType, Answer, AnswerFeedback, Phase, Scenario


class AdaptTests(TestCase):
    def setUp(self):
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.owner = User.objects.create_user('adapt_owner')
        self.owner.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='Pendulum', visibility_status='public',
            created_by=self.owner, updated_by=self.owner,
        )
        phase = Phase.objects.create(
            name='Engage', scenario=self.scenario, created_by=self.owner, updated_by=self.owner,
        )
        activity_type = ActivityType.objects.create(
            name='Question', created_by=self.owner, updated_by=self.owner,
        )
        activity = Activity.objects.create(
            name='Q1', text='Why?', scenario=self.scenario, phase=phase,
            activity_type=activity_type, created_by=self.owner, updated_by=self.owner,
        )
        self.answer = Answer.objects.create(
            activity=activity, text='Gravity', is_correct=True,
            created_by=self.owner, updated_by=self.owner,
        )
        AnswerFeedback.objects.create(
            answer=self.answer, text='Well done', vid_url='https://video.test/1',
            created_by=self.owner, updated_by=self.owner,
        )
        self.client.force_login(self.owner)
        self.url = reverse('duplicate_scenario', args=[self.scenario.id])

    def adapt(self, **data):
        return self.client.post(self.url, {'variant': 'adaptation', **data})

    def copies(self):
        return Scenario.objects.filter(origin_scenario=self.scenario)

    def test_adapt_requires_post(self):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 405)
        self.assertFalse(self.copies().exists())

    def test_uses_the_chosen_name_and_opens_the_edit_page(self):
        response = self.adapt(name='  Pendulum for year 9  ')

        copy = self.copies().get()
        self.assertEqual(copy.name, 'Pendulum for year 9')
        self.assertEqual(copy.visibility_status, 'private')
        self.assertRedirects(
            response, reverse('updateScenario', args=[copy.id]), fetch_redirect_response=False,
        )

    def test_blank_name_falls_back_to_the_automatic_one(self):
        self.adapt(name='   ')

        self.assertEqual(self.copies().get().name, 'Pendulum Copy Made by adapt_owner')

    def test_taken_name_gets_a_number(self):
        Scenario.objects.create(name='My copy', created_by=self.owner, updated_by=self.owner)

        self.adapt(name='My copy')
        self.adapt(name='My copy')

        self.assertEqual(
            sorted(self.copies().values_list('name', flat=True)), ['My copy 2', 'My copy 3'],
        )

    def test_long_name_is_capped(self):
        self.adapt(name='x' * 300)

        self.assertEqual(len(self.copies().get().name), 255)

    def test_answer_feedback_is_copied(self):
        self.adapt()

        copied_answer = Answer.objects.get(activity__scenario=self.copies().get())
        self.assertNotEqual(copied_answer.pk, self.answer.pk)
        self.assertEqual(
            list(copied_answer.feedbacks.values_list('text', 'vid_url')),
            [('Well done', 'https://video.test/1')],
        )
        self.assertEqual(self.answer.feedbacks.count(), 1)

    def test_failed_copy_leaves_nothing_behind(self):
        before = Scenario.objects.count()

        with patch('authoringtool.views.AnswerFeedback.objects.create', side_effect=RuntimeError('disk full')):
            response = self.adapt(name='Broken copy')

        self.assertEqual(Scenario.objects.count(), before)
        self.assertFalse(Phase.objects.filter(scenario__name='Broken copy').exists())
        self.assertRedirects(response, reverse('scenarios'), fetch_redirect_response=False)
        messages = [str(m) for m in response.wsgi_request._messages]
        self.assertIn("The copy couldn't be created", messages[0])

    def test_scenarios_page_has_the_adapt_modal(self):
        response = self.client.get(reverse('scenarios'))

        self.assertContains(response, 'id="adaptModal"')
        self.assertContains(response, 'Create copy')
        self.assertContains(response, 'Creating your copy')
        self.assertContains(response, f'data-action="{self.url}"')
        self.assertContains(response, 'data-default-name="Pendulum Copy Made by adapt_owner"')
        self.assertNotContains(response, f'href="{self.url}')
