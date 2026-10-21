from datetime import timedelta
from unittest.mock import patch

from django.conf import settings
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.utils import timezone

from authoringtool.machine_translation.client import (
    TranslationOutputInvalid,
    TranslationServiceUnavailable,
    TranslationTimeout,
)
from authoringtool.machine_translation.copying import copy_for_translation
from authoringtool.machine_translation.queue import (
    drain_translation_queue,
    retry_job,
)
from authoringtool.models import (
    Activity,
    Scenario,
    ScenarioTranslationJob,
    TranslationMatch,
)
from authoringtool.mt_test_fixtures import build_source_scenario, make_teacher

LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
TRANSLATE = 'authoringtool.machine_translation.queue.translate_fields'
CHECK = 'authoringtool.machine_translation.queue.check_translation_service'


def fake_translate(fields, source_language, target_language, context):
    return {key: f'EL {value}' for key, value in fields.items()}


@override_settings(CACHES=LOCMEM)
class DrainTests(TestCase):
    def setUp(self):
        cache.clear()
        self.owner = make_teacher('mt_queue_owner')
        self.source = build_source_scenario(self.owner)
        self.job = copy_for_translation(self.source, self.owner, 'Greek', 'English')

    def refresh(self, job=None):
        job = job or self.job
        job.refresh_from_db()
        return job

    def test_drain_translates_all_units_and_finishes(self):
        with patch(TRANSLATE, side_effect=fake_translate):
            drain_translation_queue()
        job = self.refresh()
        self.assertEqual(job.status, 'done')
        self.assertEqual(job.units_done, job.units_total)
        self.assertEqual(job.model, settings.LLM_MACHINE_TRANSLATION_MODEL)
        texts = Activity.objects.filter(scenario=job.target_scenario).values_list('text', flat=True)
        self.assertTrue(all(text.startswith('EL ') for text in texts))
        self.assertIsNotNone(Scenario.objects.get(pk=job.target_scenario_id).current_version_id)

    def test_finish_creates_confirmed_mt_match_with_full_map(self):
        with patch(TRANSLATE, side_effect=fake_translate):
            drain_translation_queue()
        match = TranslationMatch.objects.get()
        self.assertEqual(match.origin, 'machine_translation')
        self.assertEqual(match.status, 'confirmed')
        self.assertTrue(match.exact_match)
        self.assertEqual(len(match.activity_map), 3)
        self.assertLess(match.scenario_a_id, match.scenario_b_id)
        source_ids = set(self.source.activities.values_list('id', flat=True))
        self.assertEqual({pair[0] for pair in match.activity_map}, source_ids)
        self.assertTrue(match.structure_signature_a and match.structure_signature_b)

    def test_unavailable_requeues_and_keeps_progress(self):
        calls = {'n': 0}

        def flaky(*args):
            calls['n'] += 1
            if calls['n'] == 3:
                raise TranslationServiceUnavailable('tunnel down')
            return fake_translate(*args)

        with patch(TRANSLATE, side_effect=flaky),                 patch(CHECK, return_value={'reachable': False, 'model_present': False}):
            drain_translation_queue()
        job = self.refresh()
        self.assertEqual(job.status, 'waiting')
        self.assertEqual(job.units_done, 2)
        with patch(TRANSLATE, side_effect=fake_translate):
            drain_translation_queue()
        self.assertEqual(self.refresh().status, 'done')

    def test_invalid_output_three_times_fails_with_label(self):
        def lose_numbers(fields, *args):
            return {key: 'EL' for key in fields}

        with patch(TRANSLATE, side_effect=lose_numbers):
            drain_translation_queue()
        job = self.refresh()
        self.assertEqual(job.status, 'failed')
        self.assertTrue(job.error.startswith('Activity "'), job.error)

    def test_failed_job_never_marked_done_with_source_text(self):
        with patch(TRANSLATE, side_effect=TranslationOutputInvalid('not json')):
            drain_translation_queue()
        job = self.refresh()
        self.assertEqual(job.status, 'failed')
        self.assertEqual(job.units_done, 0)
        self.assertFalse(TranslationMatch.objects.exists())

    def test_retry_job_resumes_from_failed_unit(self):
        seen = []

        def fail_on_activities(fields, *args):
            if 'text' in fields:
                raise TranslationOutputInvalid('bad')
            return fake_translate(fields, *args)

        with patch(TRANSLATE, side_effect=fail_on_activities):
            drain_translation_queue()
        job = self.refresh()
        self.assertEqual(job.status, 'failed')
        self.assertEqual(job.units_done, 3)

        retry_job(job)
        self.assertEqual(self.refresh().status, 'waiting')
        self.assertEqual(self.refresh().error, '')

        def record(fields, *args):
            seen.append(fields)
            return fake_translate(fields, *args)

        with patch(TRANSLATE, side_effect=record):
            drain_translation_queue()
        self.assertEqual(self.refresh().status, 'done')
        self.assertEqual(len(seen), 3)
        self.assertTrue(all('text' in fields for fields in seen))

    def test_only_one_job_translating(self):
        ScenarioTranslationJob.objects.filter(pk=self.job.pk).update(
            status='translating', claimed_at=timezone.now(),
        )
        copy_for_translation(self.source, self.owner, 'Spanish', 'English')
        with patch(TRANSLATE, side_effect=fake_translate) as translate:
            drain_translation_queue()
        translate.assert_not_called()

    def test_stuck_job_released(self):
        ScenarioTranslationJob.objects.filter(pk=self.job.pk).update(
            status='translating', claimed_at=timezone.now() - timedelta(hours=4),
        )
        with patch(TRANSLATE, side_effect=fake_translate):
            drain_translation_queue()
        self.assertEqual(self.refresh().status, 'done')

    def test_jobs_run_oldest_first(self):
        second = copy_for_translation(self.source, self.owner, 'Spanish', 'English')
        order = []

        def record(fields, source, target, context):
            order.append(target)
            return fake_translate(fields, source, target, context)

        with patch(TRANSLATE, side_effect=record):
            drain_translation_queue()
        self.assertEqual(order[0], 'Greek')
        self.assertEqual(order[-1], 'Spanish')
        self.assertEqual(self.refresh(second).status, 'done')

    def test_timeout_with_service_up_counts_as_unit_failure(self):
        with patch(TRANSLATE, side_effect=TranslationTimeout('slow')), \
                patch(CHECK, return_value={'reachable': True, 'model_present': True}):
            drain_translation_queue()
        self.assertEqual(self.refresh().status, 'failed')

    def test_timeout_with_service_down_requeues(self):
        with patch(TRANSLATE, side_effect=TranslationTimeout('slow')), \
                patch(CHECK, return_value={'reachable': False, 'model_present': False}):
            drain_translation_queue()
        self.assertEqual(self.refresh().status, 'waiting')

    def test_drain_skips_job_whose_target_was_deleted(self):
        second = copy_for_translation(self.source, self.owner, 'Spanish', 'English')
        first_target_id = self.job.target_scenario_id

        def delete_first_target(fields, source, target, context):
            if target == 'Greek':
                Scenario.objects.filter(pk=first_target_id).delete()
            return fake_translate(fields, source, target, context)

        with patch(TRANSLATE, side_effect=delete_first_target):
            drain_translation_queue()
        self.assertFalse(ScenarioTranslationJob.objects.filter(pk=self.job.pk).exists())
        self.assertEqual(self.refresh(second).status, 'done')

    def test_drain_completes_scenario_with_image_only_activity(self):
        Activity.objects.filter(
            scenario=self.job.target_scenario, name='Why it halves',
        ).update(text='<p><img src="/media/images/ohm.png"></p>')
        with patch(TRANSLATE, side_effect=fake_translate):
            drain_translation_queue()
        self.assertEqual(self.refresh().status, 'done')
        self.assertEqual(
            Activity.objects.get(scenario=self.job.target_scenario, name='EL Why it halves').text,
            '<p><img src="/media/images/ohm.png"></p>',
        )

    def test_lock_prevents_overlapping_runs(self):
        cache.add('authoringtool:drain-translation-queue', '1', 60)
        with patch(TRANSLATE, side_effect=fake_translate) as translate:
            drain_translation_queue()
        translate.assert_not_called()

    def test_beat_entry_registered(self):
        entry = settings.CELERY_BEAT_SCHEDULE['drain-translation-queue']
        self.assertEqual(entry['task'], 'authoringtool.tasks.drain_translation_queue_task')
        self.assertEqual(entry['schedule'], 60.0)
