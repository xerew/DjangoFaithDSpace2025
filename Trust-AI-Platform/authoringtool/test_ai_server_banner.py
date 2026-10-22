"""The AI server (Ollama on the NTUA machine) banner on the Metrics & AI page."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import requests
from django.conf import settings
from django.contrib.auth.models import Group, User
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from .machine_translation.health import (
    check_ollama,
    proposal_server_status,
    record_service_status,
    translation_service_available,
)
from .models import Scenario

LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
GET = 'authoringtool.machine_translation.health.requests.get'
PROPOSAL_MODELS = [
    settings.LLM_PROPOSAL_MODEL,
    settings.LLM_IMAGE_DESCRIPTION_MODEL,
    settings.LLM_TRANSLATION_MODEL,
]


def tags(*names):
    response = MagicMock(status_code=200)
    response.json.return_value = {'models': [{'name': name} for name in names]}
    return response


def server_up():
    with patch(GET, return_value=tags(*PROPOSAL_MODELS, settings.LLM_MACHINE_TRANSLATION_MODEL)):
        check_ollama()


def server_down():
    with patch(GET, side_effect=requests.ConnectionError('Connection refused')):
        check_ollama()


@override_settings(CACHES=LOCMEM)
class ProposalServerStatusTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_ok_when_all_proposal_models_present(self):
        server_up()

        status = proposal_server_status()

        self.assertEqual(status['state'], 'ok')
        self.assertEqual(status['missing'], [])

    def test_missing_models_are_listed(self):
        with patch(GET, return_value=tags(settings.LLM_IMAGE_DESCRIPTION_MODEL)):
            check_ollama()

        status = proposal_server_status()

        self.assertEqual(status['state'], 'missing_models')
        self.assertEqual(
            status['missing'], [settings.LLM_PROPOSAL_MODEL, settings.LLM_TRANSLATION_MODEL],
        )
        self.assertIn(settings.LLM_PROPOSAL_MODEL, status['title'])

    def test_offline_since_is_kept_across_checks_and_cleared_when_back(self):
        first = timezone.now()
        with patch('authoringtool.machine_translation.health.timezone.now', return_value=first):
            server_down()
        with patch(
            'authoringtool.machine_translation.health.timezone.now',
            return_value=first + timedelta(minutes=2),
        ):
            server_down()

        status = proposal_server_status(now=first + timedelta(minutes=2))
        self.assertEqual(status['state'], 'offline')
        self.assertEqual(status['offline_since'], first)
        self.assertEqual(status['checked_at'], first + timedelta(minutes=2))
        self.assertIn('Offline since', status['detail'])
        self.assertIn('every minute', status['detail'])

        server_up()
        self.assertIsNone(proposal_server_status()['offline_since'])

    def test_stale_or_missing_status_counts_as_offline(self):
        self.assertEqual(proposal_server_status()['state'], 'offline')

        server_up()
        later = timezone.now() + timedelta(seconds=181)
        self.assertEqual(proposal_server_status(now=later)['state'], 'offline')

    def test_translation_check_still_works(self):
        record_service_status(True, True)
        self.assertTrue(translation_service_available())
        self.assertEqual(proposal_server_status()['state'], 'missing_models')


@override_settings(CACHES=LOCMEM)
class MetricsPageBannerTests(TestCase):
    def setUp(self):
        cache.clear()
        teachers, _ = Group.objects.get_or_create(name='teachers')
        self.owner = User.objects.create_user('banner_owner')
        self.owner.groups.add(teachers)
        self.scenario = Scenario.objects.create(
            name='Banner scenario', visibility_status='private',
            created_by=self.owner, updated_by=self.owner,
        )
        self.client.force_login(self.owner)
        self.metrics_url = reverse('ai_metrics', args=[self.scenario.id])
        self.generate_url = reverse('generate_llm_context', args=[self.scenario.id])

    def test_banner_and_disabled_buttons_when_offline(self):
        server_down()

        response = self.client.get(self.metrics_url)

        self.assertContains(response, 'The AI server is offline')
        self.assertContains(response, 'Check again')
        content = response.content.decode()
        for button_id in ('btn-generate-context', 'btn-generate-context-force'):
            tag = content[content.index(f'id="{button_id}"'):]
            tag = tag[:tag.index('>')]
            self.assertIn('disabled', tag)

    def test_no_banner_when_server_ok(self):
        server_up()

        response = self.client.get(self.metrics_url)

        self.assertNotContains(response, 'The AI server is offline')
        content = response.content.decode()
        tag = content[content.index('id="btn-generate-context"'):]
        self.assertNotIn('disabled', tag[:tag.index('>')])

    def test_generation_is_refused_while_offline(self):
        server_down()

        with patch('authoringtool.views.generate_llm_context_for_scenario.delay') as delay:
            response = self.client.post(self.generate_url)

        self.assertEqual(response.status_code, 503)
        self.assertIn('offline', response.json()['error'])
        delay.assert_not_called()

    def test_generation_runs_when_server_ok(self):
        server_up()

        with patch('authoringtool.views.generate_llm_context_for_scenario.delay') as delay:
            delay.return_value.id = 'task-1'
            response = self.client.post(self.generate_url)

        self.assertEqual(response.json(), {'status': 'started', 'task_id': 'task-1'})

    def test_status_endpoint_reports_banner_text(self):
        server_down()

        response = self.client.get(reverse('ai_server_status'))

        data = response.json()
        self.assertEqual(data['state'], 'offline')
        self.assertIn('The AI server is offline', data['title'])
        self.assertIn('Offline since', data['detail'])

    def test_check_again_asks_the_server_now(self):
        server_down()

        with patch(GET, return_value=tags(*PROPOSAL_MODELS)):
            response = self.client.post(reverse('ai_server_check'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(proposal_server_status()['state'], 'ok')

    def test_check_again_requires_post_and_a_teacher(self):
        self.assertEqual(self.client.get(reverse('ai_server_check')).status_code, 405)
        self.client.force_login(User.objects.create_user('banner_student'))
        self.assertNotEqual(self.client.post(reverse('ai_server_check')).status_code, 200)
