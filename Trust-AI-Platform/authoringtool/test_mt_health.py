from datetime import timedelta
from unittest.mock import MagicMock, patch

import requests
from django.conf import settings
from django.core.cache import cache
from django.test import SimpleTestCase, override_settings
from django.utils import timezone

from authoringtool.machine_translation.health import (
    check_translation_service,
    record_service_status,
    translation_service_available,
)

LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
GET = 'authoringtool.machine_translation.health.requests.get'


def tags(*names):
    response = MagicMock(status_code=200)
    response.json.return_value = {'models': [{'name': name} for name in names]}
    return response


@override_settings(CACHES=LOCMEM)
class HealthTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_available_when_model_listed(self):
        with patch(GET, return_value=tags('qwen3.6:35b', settings.LLM_MACHINE_TRANSLATION_MODEL)) as get:
            status = check_translation_service()
        self.assertTrue(status['reachable'] and status['model_present'])
        self.assertEqual(get.call_args.kwargs['timeout'], 3)
        self.assertTrue(translation_service_available())

    def test_unavailable_when_model_missing(self):
        with patch(GET, return_value=tags('qwen3.6:35b')):
            status = check_translation_service()
        self.assertTrue(status['reachable'])
        self.assertFalse(status['model_present'])
        self.assertFalse(translation_service_available())

    def test_unavailable_on_connection_error(self):
        with patch(GET, side_effect=requests.ConnectionError('down')):
            status = check_translation_service()
        self.assertFalse(status['reachable'])
        self.assertFalse(translation_service_available())

    def test_unavailable_when_status_missing(self):
        self.assertFalse(translation_service_available())

    def test_unavailable_when_status_stale(self):
        record_service_status(True, True)
        later = timezone.now() + timedelta(seconds=181)
        self.assertTrue(translation_service_available())
        self.assertFalse(translation_service_available(now=later))

    def test_beat_entry_registered(self):
        entry = settings.CELERY_BEAT_SCHEDULE['check-translation-service']
        self.assertEqual(entry['task'], 'authoringtool.tasks.check_translation_service_task')
        self.assertEqual(entry['schedule'], 60.0)
