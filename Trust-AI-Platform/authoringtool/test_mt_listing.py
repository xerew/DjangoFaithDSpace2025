import html
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from django.utils.html import strip_tags

from authoringtool.machine_translation.copying import copy_for_translation
from authoringtool.models import Scenario, ScenarioTranslationJob, TranslationMatch
from authoringtool.mt_test_fixtures import build_source_scenario, make_teacher

AVAILABLE = 'authoringtool.views.translation_service_available'
DELAY = 'authoringtool.tasks.drain_translation_queue_task.delay'


class ListingTests(TestCase):
    def setUp(self):
        self.owner = make_teacher('mt_list_owner')
        self.requester = make_teacher('mt_list_requester')
        self.reviewer = make_teacher('mt_list_reviewer')
        self.source = build_source_scenario(self.owner)
        self.job = copy_for_translation(self.source, self.requester, 'Greek', 'English')
        self.target = self.job.target_scenario
        self.client.force_login(self.requester)

    def set_job(self, **fields):
        ScenarioTranslationJob.objects.filter(pk=self.job.pk).update(**fields)
        self.job.refresh_from_db()

    def page(self, url=None):
        with patch(AVAILABLE, return_value=True):
            response = self.client.get(url or reverse('scenarios'))
        return html.unescape(response.content.decode())

    def card_title(self, scenario):
        return f'sc-card-title">{scenario.name}<'

    def confirm_pair(self):
        a, b = sorted([self.source, self.target], key=lambda s: s.id)
        TranslationMatch.objects.create(
            scenario_a=a, scenario_b=b, status='confirmed',
            origin='machine_translation', exact_match=True,
        )

    def test_card_shows_progress_badge(self):
        self.set_job(status='translating', completed_units=['scenario', 'phase:1'])
        self.assertIn('Translating… 2 / 6', self.page())

    def test_card_shows_waiting_in_queue(self):
        self.assertIn('Waiting in queue', self.page())

    def test_waiting_tag_says_service_unavailable_when_down(self):
        with patch(AVAILABLE, return_value=False):
            content = html.unescape(self.client.get(reverse('scenarios')).content.decode())
        self.assertIn('Waiting — translation service unavailable', content)

    def test_card_shows_machine_translated_tag(self):
        self.set_job(status='done')
        self.assertIn('Machine-translated', self.page())

    def test_card_shows_verified_tag_with_name(self):
        self.set_job(status='done', verified_by=self.reviewer, verified_at=timezone.now())
        content = self.page()
        self.assertIn('Translation verified', content)
        self.assertIn('Verified by mt_list_reviewer', content)

    def test_failed_card_has_retry(self):
        self.set_job(status='failed', error='Activity "X": changed the numbers')
        content = self.page()
        self.assertIn('Translation failed', content)
        self.assertIn(reverse('retry_machine_translation', args=[self.job.id]), content)

    def test_pending_card_hides_view_and_edit(self):
        content = self.page()
        self.assertNotIn(reverse('viewScenario', args=[self.target.id]) + '"', content)
        self.assertNotIn(reverse('updateScenario', args=[self.target.id]) + '"', content)
        self.set_job(status='done')
        self.assertIn(reverse('viewScenario', args=[self.target.id]) + '"', self.page())

    def test_private_translation_not_bundled_into_original_card(self):
        self.set_job(status='done')
        self.confirm_pair()
        content = self.page()
        self.assertIn(self.card_title(self.target), content)
        self.assertIn(self.card_title(self.source), content)

    def test_published_translation_bundles(self):
        self.set_job(status='done')
        self.confirm_pair()
        Scenario.objects.filter(pk=self.target.pk).update(visibility_status='public')
        content = self.page()
        self.assertNotIn(self.card_title(self.target), content)
        self.assertIn('Machine-translated', content)  # tag in the Translations modal

    def test_ready_message_shown_once(self):
        self.set_job(status='done')
        content = self.page()
        self.assertIn(f'Your Greek translation of {self.target.name} is ready.', strip_tags(content))
        self.assertIn(f'<a href="{reverse("viewScenario", args=[self.target.id])}">', content)
        self.assertNotIn('is ready.', self.page())

    def test_failed_message_shown(self):
        self.set_job(status='failed')
        self.assertIn(
            f'Your Greek translation of {self.target.name} failed. Open your scenarios to retry.',
            self.page(),
        )

    def test_no_message_for_other_users(self):
        self.set_job(status='done')
        self.client.force_login(self.owner)
        self.assertNotIn('is ready.', self.page())
        self.job.refresh_from_db()
        self.assertIsNone(self.job.notified_at)

    def test_job_status_endpoint_reports_progress(self):
        self.set_job(status='translating', completed_units=['scenario'])
        response = self.client.get(reverse('machine_translation_job_status'), {'ids': str(self.job.id)})
        self.assertEqual(
            response.json()['jobs'][str(self.job.id)],
            {'status': 'translating', 'units_done': 1, 'units_total': 6},
        )

    def test_job_status_endpoint_hides_other_users_jobs(self):
        self.client.force_login(self.owner)
        response = self.client.get(reverse('machine_translation_job_status'), {'ids': str(self.job.id)})
        self.assertEqual(response.json()['jobs'], {})

    def test_retry_requeues_failed_job(self):
        self.set_job(status='failed', error='boom')
        with patch(DELAY) as delay, self.captureOnCommitCallbacks(execute=True):
            response = self.client.post(reverse('retry_machine_translation', args=[self.job.id]))
        self.assertEqual(response.status_code, 302)
        self.job.refresh_from_db()
        self.assertEqual(self.job.status, 'waiting')
        delay.assert_called_once()

    def test_other_teacher_cannot_retry(self):
        self.set_job(status='failed')
        self.client.force_login(self.owner)
        response = self.client.post(reverse('retry_machine_translation', args=[self.job.id]))
        self.assertEqual(response.status_code, 403)

    def test_scenario_page_shows_tag(self):
        self.set_job(status='done')
        self.assertIn('Machine-translated', self.page(reverse('viewScenario', args=[self.target.id])))
