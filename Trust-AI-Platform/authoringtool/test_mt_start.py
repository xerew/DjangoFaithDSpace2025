import html
import re
from unittest.mock import patch

from django.conf import settings
from django.test import TestCase, override_settings
from django.urls import reverse

from authoringtool.models import Language, Scenario, ScenarioTranslationJob
from authoringtool.mt_test_fixtures import build_source_scenario, make_teacher

AVAILABLE = 'authoringtool.views.translation_service_available'
DELAY = 'authoringtool.tasks.drain_translation_queue_task.delay'
BANNER = (
    "The translation service can't be reached right now. Machine translation "
    "is paused. Please try again later."
)
EMOJI = re.compile('[\U0001F300-\U0001FAFF☀-➿]')


# These cover the flow once machine translation is rolled out to teachers.
@override_settings(MACHINE_TRANSLATION_FOR_ALL=True)
class StartTranslationTests(TestCase):
    def setUp(self):
        Language.objects.get_or_create(name='Greek')
        Language.objects.get_or_create(name='English')
        self.owner = make_teacher('mt_start_owner')
        self.teacher = make_teacher('mt_start_teacher')
        self.source = build_source_scenario(self.owner)
        self.client.force_login(self.teacher)

    def start(self, scenario=None, **data):
        scenario = scenario or self.source
        return self.client.post(
            reverse('start_machine_translation', args=[scenario.id]),
            {'target_language': 'Greek', **data},
        )

    def test_teacher_can_start_translation(self):
        with patch(AVAILABLE, return_value=True), patch(DELAY) as delay, \
                self.captureOnCommitCallbacks(execute=True):
            response = self.start()
        self.assertRedirects(response, reverse('scenarios'), fetch_redirect_response=False)
        job = ScenarioTranslationJob.objects.get()
        self.assertEqual(job.target_scenario.visibility_status, 'private')
        self.assertEqual(job.target_scenario.created_by, self.teacher)
        self.assertEqual((job.source_language, job.target_language), ('English', 'Greek'))
        delay.assert_called_once()

    def test_refused_when_service_unavailable(self):
        with patch(AVAILABLE, return_value=False), patch(DELAY):
            response = self.start()
        self.assertFalse(ScenarioTranslationJob.objects.exists())
        messages = [str(m) for m in response.wsgi_request._messages]
        self.assertIn(BANNER, messages)

    def test_refused_when_target_equals_source(self):
        with patch(AVAILABLE, return_value=True), patch(DELAY):
            self.start(target_language='English')
        self.assertFalse(ScenarioTranslationJob.objects.exists())

    def test_refused_for_unknown_language(self):
        with patch(AVAILABLE, return_value=True), patch(DELAY):
            self.start(target_language='Klingon')
        self.assertFalse(ScenarioTranslationJob.objects.exists())

    def test_refused_when_user_cannot_view_scenario(self):
        Scenario.objects.filter(pk=self.source.pk).update(visibility_status='private')
        with patch(AVAILABLE, return_value=True), patch(DELAY):
            response = self.start()
        self.assertEqual(response.status_code, 403)
        self.assertFalse(ScenarioTranslationJob.objects.exists())

    def test_start_requires_source_language_when_scenario_has_none(self):
        Scenario.objects.filter(pk=self.source.pk).update(language='')
        with patch(AVAILABLE, return_value=True), patch(DELAY):
            self.start()
            self.assertFalse(ScenarioTranslationJob.objects.exists())
            self.start(source_language='Greek')
            self.assertFalse(ScenarioTranslationJob.objects.exists())
            self.start(source_language='English')
        self.assertEqual(ScenarioTranslationJob.objects.get().source_language, 'English')

    def test_refused_while_source_is_itself_pending(self):
        with patch(AVAILABLE, return_value=True), patch(DELAY):
            self.start()
            pending = ScenarioTranslationJob.objects.get().target_scenario
            Scenario.objects.filter(pk=pending.pk).update(visibility_status='public')
            self.start(scenario=pending, target_language='English')
        self.assertEqual(ScenarioTranslationJob.objects.count(), 1)

    def test_get_not_allowed(self):
        response = self.client.get(reverse('start_machine_translation', args=[self.source.id]))
        self.assertEqual(response.status_code, 405)

    def test_status_endpoint_reports_availability(self):
        with patch(AVAILABLE, return_value=False):
            response = self.client.get(reverse('machine_translation_status'))
        self.assertEqual(response.json(), {'available': False})
        with patch(AVAILABLE, return_value=True):
            response = self.client.get(reverse('machine_translation_status'))
        self.assertEqual(response.json(), {'available': True})

    def test_scenarios_page_has_translate_modal_without_emojis(self):
        response = self.client.get(reverse('scenarios'))
        content = html.unescape(response.content.decode())
        self.assertIn('Start translation', content)
        self.assertIn(BANNER, content)
        self.assertIn(reverse('start_machine_translation', args=[self.source.id]), content)
        self.assertIsNone(EMOJI.search(content))


class AdminOnlyRolloutTests(TestCase):
    """Until MACHINE_TRANSLATION_FOR_ALL is switched on, only admins translate."""

    def setUp(self):
        Language.objects.get_or_create(name='Greek')
        self.owner = make_teacher('mt_gate_owner')
        self.teacher = make_teacher('mt_gate_teacher')
        self.admin = make_teacher('mt_gate_admin', admin=True)
        self.source = build_source_scenario(self.owner)
        self.url = reverse('start_machine_translation', args=[self.source.id])

    def test_setting_is_off_by_default(self):
        self.assertFalse(settings.MACHINE_TRANSLATION_FOR_ALL)

    def test_teacher_cannot_start_translation(self):
        self.client.force_login(self.teacher)
        with patch(AVAILABLE, return_value=True), patch(DELAY):
            response = self.client.post(self.url, {'target_language': 'Greek'})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(ScenarioTranslationJob.objects.exists())

    def test_admin_can_start_translation(self):
        self.client.force_login(self.admin)
        with patch(AVAILABLE, return_value=True), patch(DELAY):
            self.client.post(self.url, {'target_language': 'Greek'})
        self.assertEqual(ScenarioTranslationJob.objects.get().requested_by, self.admin)

    def test_translate_button_shown_to_admins_only(self):
        self.client.force_login(self.teacher)
        content = self.client.get(reverse('scenarios')).content.decode()
        self.assertNotIn(self.url, content)
        self.assertNotIn('id="translateModal"', content)
        self.client.force_login(self.admin)
        content = self.client.get(reverse('scenarios')).content.decode()
        self.assertIn(self.url, content)
        self.assertIn('id="translateModal"', content)
