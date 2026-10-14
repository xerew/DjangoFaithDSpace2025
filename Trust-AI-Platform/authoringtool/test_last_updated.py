"""Scenarios, phases and activities show when and by whom they were last
updated, and editing something inside them counts as an update."""
from datetime import timedelta

from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from authoringtool.models import Activity, ActivityType, Answer, Phase, Scenario


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class LastUpdatedTests(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user(
            'lu_owner', password='pass', first_name='Olga', last_name='Owner',
        )
        self.editor = User.objects.create_user(
            'lu_editor', password='pass', first_name='Eddie', last_name='Editor',
            is_staff=True,
        )
        for user in (self.owner, self.editor):
            user.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='Timeline', created_by=self.owner, updated_by=self.owner,
        )
        self.phase = Phase.objects.create(
            name='Explore', scenario=self.scenario, created_by=self.owner, updated_by=self.owner,
        )
        self.second_phase = Phase.objects.create(
            name='Explain', scenario=self.scenario, created_by=self.owner, updated_by=self.owner,
        )
        question = ActivityType.objects.create(
            name='Question', created_by=self.owner, updated_by=self.owner,
        )
        self.activity = Activity.objects.create(
            name='Q1', text='Which?', phase=self.phase, scenario=self.scenario,
            activity_type=question, created_by=self.owner, updated_by=self.owner,
        )
        self.answer = Answer.objects.create(
            activity=self.activity, text='A', is_correct=True,
            created_by=self.owner, updated_by=self.owner,
        )
        # Pretend everything was last touched a week ago.
        week_ago = timezone.now() - timedelta(days=7)
        for model in (Scenario, Phase, Activity):
            model.objects.update(updated_on=week_ago)
        self.week_ago = week_ago

    def assertTouchedBy(self, obj, user):
        obj.refresh_from_db()
        self.assertGreater(obj.updated_on, self.week_ago + timedelta(days=6))
        self.assertEqual(obj.updated_by, user)

    def test_pages_show_last_updated_and_by_whom(self):
        self.client.force_login(self.owner)
        pages = {
            'scenario': reverse('viewScenario', args=[self.scenario.id]),
            'phase': reverse('phase', args=[self.scenario.id, self.phase.id]),
            'activity': reverse('activity', args=[self.scenario.id, self.phase.id, self.activity.id]),
        }
        for name, url in pages.items():
            with self.subTest(page=name):
                response = self.client.get(url)
                self.assertContains(response, 'Last updated')
                self.assertContains(response, 'Olga Owner')

    def test_editing_answers_updates_activity_phase_and_scenario(self):
        self.client.force_login(self.editor)
        self.client.post(
            reverse('updateAnswers', args=[self.scenario.id, self.phase.id, self.activity.id]),
            {
                'answers[0][id]': self.answer.id, 'answers[0][text]': 'A (edited)',
                'answers[0][is_correct]': 'on', 'answers[0][answer_weight]': 3,
            },
        )
        for obj in (self.activity, self.phase, self.scenario):
            with self.subTest(model=type(obj).__name__):
                self.assertTouchedBy(obj, self.editor)

    def test_reordering_phases_updates_the_scenario(self):
        self.client.force_login(self.editor)
        self.client.post(reverse('move_phase', args=[self.scenario.id, self.second_phase.id, 'up']))
        self.assertTouchedBy(self.scenario, self.editor)

    def test_deleting_an_activity_updates_phase_and_scenario(self):
        self.client.force_login(self.editor)
        self.client.post(reverse(
            'deleteActivity', args=[self.scenario.id, self.phase.id, self.activity.id],
        ))
        self.assertTouchedBy(self.phase, self.editor)
        self.assertTouchedBy(self.scenario, self.editor)

    def test_only_viewing_changes_nothing(self):
        self.client.force_login(self.editor)
        self.client.get(reverse('viewScenario', args=[self.scenario.id]))
        self.scenario.refresh_from_db()
        self.assertEqual(self.scenario.updated_by, self.owner)
        self.assertLess(self.scenario.updated_on, timezone.now() - timedelta(days=6))
