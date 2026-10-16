"""The activity screen links to the activity's delete confirmation."""
from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.models import Activity, ActivityType, Phase, Scenario


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class ActivityDeleteButtonTests(TestCase):
    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('delete_owner', password='pass')
        self.other = User.objects.create_user('delete_other', password='pass')
        for user in (self.owner, self.other):
            user.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='Deletable', visibility_status='public',
            created_by=self.owner, updated_by=self.owner,
        )
        self.phase = Phase.objects.create(
            name='Explore', scenario=self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )
        kind = ActivityType.objects.create(
            name='Explanation', created_by=self.owner, updated_by=self.owner,
        )
        self.activity = Activity.objects.create(
            name='Old hint', text='Hint', phase=self.phase,
            scenario=self.scenario, activity_type=kind,
            created_by=self.owner, updated_by=self.owner,
        )
        self.page_url = reverse(
            'activity', args=[self.scenario.id, self.phase.id, self.activity.id],
        )
        self.delete_url = reverse(
            'deleteActivity',
            args=[self.scenario.id, self.phase.id, self.activity.id],
        )

    def test_editor_sees_delete_button(self):
        self.client.force_login(self.owner)

        response = self.client.get(self.page_url)

        self.assertContains(response, f'href="{self.delete_url}"')
        self.assertContains(response, 'Delete')

    def test_teacher_who_cannot_edit_sees_no_delete_button(self):
        self.client.force_login(self.other)

        response = self.client.get(self.page_url)

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, f'href="{self.delete_url}"')

    def test_button_leads_to_confirmation_that_returns_to_the_activity(self):
        self.client.force_login(self.owner)

        response = self.client.get(self.delete_url)

        self.assertContains(response, 'Delete activity')
        self.assertContains(response, f'href="{self.page_url}"')
        self.assertTrue(Activity.objects.filter(pk=self.activity.pk).exists())

    def test_confirmation_sits_below_the_fixed_top_bar(self):
        self.client.force_login(self.owner)

        response = self.client.get(self.delete_url)

        # Pages clear the fixed header and sidebar through <main class="main">.
        self.assertContains(response, '<main id="main" class="main">')
