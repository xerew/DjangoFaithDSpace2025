"""Authoring tool filters are kept in the user's session, so they survive a
refresh and coming back to the page."""
import json
from urllib.parse import parse_qs, urlparse

from django.contrib.auth.models import Group, User
from django.test import TestCase
from django.urls import reverse

from authoringtool.models import Scenario
from usergroups.models import UserGroup


class ScenarioListFilterTests(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user('filters_teacher', password='pass')
        self.teacher.groups.add(Group.objects.create(name='teachers'))
        self.client.force_login(self.teacher)
        self.url = reverse('scenarios')

    def test_filters_are_restored_when_coming_back(self):
        self.client.get(self.url, {'q': 'pend', 'language': 'Greek', 'visibility': 'all'})
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        restored = parse_qs(urlparse(response['Location']).query)
        self.assertEqual(restored['q'], ['pend'])
        self.assertEqual(restored['language'], ['Greek'])

    def test_multi_value_filters_are_restored(self):
        self.client.get(self.url, {'subject': ['3', '5']})
        restored = parse_qs(urlparse(self.client.get(self.url)['Location']).query)
        self.assertEqual(restored['subject'], ['3', '5'])

    def test_reset_forgets_the_filters(self):
        self.client.get(self.url, {'q': 'pend'})
        self.client.get(self.url, {
            'q': '', 'start_date': '', 'end_date': '', 'language': '', 'visibility': 'all',
        })
        self.assertEqual(self.client.get(self.url).status_code, 200)

    def test_paging_alone_neither_restores_nor_forgets(self):
        self.client.get(self.url, {'q': 'pend'})
        self.assertEqual(self.client.get(self.url, {'page': '2'}).status_code, 200)
        self.assertEqual(self.client.get(self.url).status_code, 302)


class DashboardFilterTests(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user('dash_teacher', password='pass')
        self.teacher.groups.add(Group.objects.create(name='teachers'))
        self.client.force_login(self.teacher)
        self.scenario = Scenario.objects.create(
            name='Dash', created_by=self.teacher, updated_by=self.teacher,
        )
        self.group = UserGroup.objects.create(
            name='Dash class', prefix='dsh', number_of_users=0, created_by=self.teacher,
        )

    def save(self, **data):
        return self.client.post(reverse('save_dashboard_filters'), data)

    def saved(self):
        response = self.client.get(reverse('index'))
        return json.loads(response.context['dashboard_filters_json'])

    def test_applied_filters_come_back_on_the_dashboard(self):
        response = self.save(
            scenario_id=self.scenario.id, start_date='2026-09-01',
            end_date='2026-10-01', group_ids=[self.group.id],
        )
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.saved(), {
            'scenario_id': self.scenario.id, 'start_date': '2026-09-01',
            'end_date': '2026-10-01', 'group_ids': [self.group.id],
        })

    def test_invalid_or_foreign_values_are_dropped(self):
        stranger = User.objects.create_user('dash_stranger')
        private = Scenario.objects.create(
            name='Private', visibility_status='private',
            created_by=stranger, updated_by=stranger,
        )
        other_group = UserGroup.objects.create(
            name='Other', prefix='oth', number_of_users=0, created_by=stranger,
        )
        self.save(scenario_id=private.id, start_date='not-a-date', group_ids=[other_group.id])
        self.assertEqual(self.saved(), {})

    def test_reset_clears_the_saved_filters(self):
        self.save(scenario_id=self.scenario.id)
        self.save(scenario_id='')
        self.assertEqual(self.saved(), {})

    def test_requires_post(self):
        self.assertEqual(self.client.get(reverse('save_dashboard_filters')).status_code, 405)
