from django.conf import settings
from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import UserProfile
from authoringtool.models import Activity, ActivityType, Phase, Scenario
from usergroups.models import UserGroup, UserGroupMembership


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class ScenarioLanguageTests(TestCase):
    """The scenario page uses the scenario's language; other pages the user's."""

    def setUp(self):
        teachers = Group.objects.create(name='teachers')
        self.owner = User.objects.create_user('lang_owner', password='pass')
        self.owner.groups.add(teachers)
        self.student = User.objects.create_user('lang_student', password='pass')
        self.scenario = Scenario.objects.create(
            name='Language scenario', visibility_status='private', language='Greek',
            created_by=self.owner, updated_by=self.owner,
        )
        phase = Phase.objects.create(
            name='Phase', scenario=self.scenario,
            created_by=self.owner, updated_by=self.owner,
        )
        activity_type, _created = ActivityType.objects.get_or_create(
            name='Explanation',
            defaults={'created_by': self.owner, 'updated_by': self.owner},
        )
        self.scenario.first_activity = Activity.objects.create(
            name='Intro', text='<p>Hello</p>', phase=phase,
            scenario=self.scenario, activity_type=activity_type,
            created_by=self.owner, updated_by=self.owner,
        )
        group = UserGroup.objects.create(
            name='Class A', prefix='cla', number_of_users=1, created_by=self.owner,
        )
        UserGroupMembership.objects.create(group=group, user=self.student)
        group.assigned_scenarios.add(self.scenario)
        UserProfile.objects.update_or_create(user=self.student, defaults={'language': 'de'})
        self.client.force_login(self.student)
        self.client.cookies[settings.LANGUAGE_COOKIE_NAME] = 'de'

    def open_scenario(self, language):
        self.scenario.language = language
        self.scenario.save(update_fields=['language'])
        return self.client.get(reverse('studentView', args=[self.scenario.id]))

    def test_scenario_page_follows_scenario_language(self):
        response = self.open_scenario('Greek')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['LANGUAGE_CODE'], 'el')
        self.assertContains(response, '<html lang="el"')

    def test_scenario_language_does_not_change_the_users_language(self):
        response = self.open_scenario('Greek')

        self.assertEqual(UserProfile.objects.get(user=self.student).language, 'de')
        cookie = response.cookies.get(settings.LANGUAGE_COOKIE_NAME)
        self.assertTrue(cookie is None or cookie.value == 'de')
        after = self.client.get(reverse('studentScenarios'))
        self.assertEqual(after.context['LANGUAGE_CODE'], 'de')

    def test_scenario_language_matching_is_lenient(self):
        response = self.open_scenario(' greek ')
        self.assertEqual(response.context['LANGUAGE_CODE'], 'el')

        response = self.open_scenario('Klingon')
        self.assertEqual(response.context['LANGUAGE_CODE'], 'de')

    def test_scenario_without_a_language_uses_the_users_language(self):
        response = self.open_scenario('')
        self.assertEqual(response.context['LANGUAGE_CODE'], 'de')

    def test_scenario_list_follows_user_language(self):
        response = self.client.get(reverse('studentScenarios'))

        self.assertEqual(response.context['LANGUAGE_CODE'], 'de')
        self.assertContains(response, '<html lang="de"')
