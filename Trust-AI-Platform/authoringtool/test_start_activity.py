from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.models import (
    Activity,
    ActivityType,
    NextQuestionLogic,
    Phase,
    Scenario,
    UserScenarioScore,
)


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class MakeFirstActivityTests(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('first_owner', password='pass')
        self.other = User.objects.create_user('first_other', password='pass')
        for teacher in (self.owner, self.other):
            teacher.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='Pendulum', visibility_status='public',
            created_by=self.owner, updated_by=self.owner,
        )
        self.phase = Phase.objects.create(
            name='Engage', scenario=self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )
        self.explanation = ActivityType.objects.create(
            name='Explanation', created_by=self.owner, updated_by=self.owner,
        )
        self.original_first = self.activity('Welcome')
        self.new_intro = self.activity('New introduction')
        self.scenario.refresh_from_db()
        self.client.force_login(self.owner)

    def activity(self, name, scenario=None, phase=None):
        return Activity.objects.create(
            name=name, text=name, scenario=scenario or self.scenario,
            phase=phase or self.phase, activity_type=self.explanation,
            created_by=self.owner, updated_by=self.owner,
        )

    def make_first(self, activity, follow=True):
        return self.client.post(
            reverse('set_start_activity', args=[self.scenario.id, activity.id]),
            follow=follow,
        )

    def activity_page(self, activity):
        return self.client.get(
            reverse('activity', args=[self.scenario.id, self.phase.id, activity.id]),
        )

    def test_owner_can_make_an_activity_the_first(self):
        NextQuestionLogic.objects.create(
            activity=self.new_intro, next_activity=self.original_first,
        )

        response = self.make_first(self.new_intro)

        self.scenario.refresh_from_db()
        self.assertEqual(self.scenario.start_activity, self.new_intro)
        self.assertRedirects(
            response,
            reverse('activity', args=[self.scenario.id, self.phase.id, self.new_intro.id]),
        )
        self.assertContains(response, 'now the first activity')
        self.assertNotContains(response, 'does not lead to another activity')

    def test_warns_when_the_new_first_activity_leads_nowhere(self):
        response = self.make_first(self.new_intro)

        self.scenario.refresh_from_db()
        self.assertEqual(self.scenario.start_activity, self.new_intro)
        self.assertContains(response, 'does not lead to another activity')

    def test_activity_from_another_scenario_is_refused(self):
        other = Scenario.objects.create(
            name='Other', created_by=self.owner, updated_by=self.owner,
        )
        foreign = self.activity('Foreign', scenario=other)

        response = self.make_first(foreign, follow=False)

        self.assertEqual(response.status_code, 404)
        self.scenario.refresh_from_db()
        self.assertEqual(self.scenario.start_activity, self.original_first)

    def test_other_teacher_cannot_change_the_first_activity(self):
        self.client.force_login(self.other)

        response = self.make_first(self.new_intro, follow=False)

        self.assertEqual(response.status_code, 403)
        self.scenario.refresh_from_db()
        self.assertEqual(self.scenario.start_activity, self.original_first)

    def test_requires_post(self):
        response = self.client.get(
            reverse('set_start_activity', args=[self.scenario.id, self.new_intro.id]),
        )

        self.assertEqual(response.status_code, 405)

    def test_activity_page_offers_the_button(self):
        response = self.activity_page(self.new_intro)

        self.assertContains(response, 'Make this the first activity')

    def test_first_activity_page_shows_a_badge_instead(self):
        response = self.activity_page(self.original_first)

        self.assertContains(response, 'First activity')
        self.assertNotContains(response, 'Make this the first activity')

    def test_other_teacher_does_not_see_the_button(self):
        self.client.force_login(self.other)

        response = self.activity_page(self.new_intro)

        self.assertNotContains(response, 'Make this the first activity')

    @override_settings(SCENARIO_REVISION_PROTECTION=True)
    def test_revision_protection_still_applies_when_switched_on(self):
        student = User.objects.create_user('first_student')
        UserScenarioScore.objects.create(user=student, scenario=self.scenario)

        self.make_first(self.new_intro)

        self.scenario.refresh_from_db()
        self.assertEqual(self.scenario.start_activity, self.original_first)
