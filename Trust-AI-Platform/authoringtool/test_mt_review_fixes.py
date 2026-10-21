"""Regression tests for the whole-branch review findings."""

import shutil
import tempfile
from datetime import timedelta
from unittest.mock import patch

from django.core.cache import cache
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from authoringtool.machine_translation.client import TranslationServiceUnavailable
from authoringtool.machine_translation.copying import copy_for_translation
from authoringtool.machine_translation.queue import LOCK_KEY, drain_translation_queue
from authoringtool.machine_translation.validation import validate_unit
from authoringtool.models import Activity, Answer, Scenario, ScenarioTranslationJob
from authoringtool.mt_test_fixtures import build_source_scenario, make_teacher

LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
TRANSLATE = 'authoringtool.machine_translation.queue.translate_fields'
CHECK = 'authoringtool.machine_translation.queue.check_translation_service'
UP = {'reachable': True, 'model_present': True}


def fake_translate(fields, source_language, target_language, context):
    return {key: f'EL {value}' for key, value in fields.items()}


@override_settings(CACHES=LOCMEM)
class QueueReviewFixTests(TestCase):
    def setUp(self):
        cache.clear()
        self.owner = make_teacher('mt_fix_owner')
        self.source = build_source_scenario(self.owner)
        self.job = copy_for_translation(self.source, self.owner, 'Greek', 'English')

    def status(self, job=None):
        job = job or self.job
        job.refresh_from_db()
        return job.status

    # C1 -------------------------------------------------------------------
    def test_unexpected_error_fails_job_visibly_and_frees_queue(self):
        second = copy_for_translation(self.source, self.owner, 'Spanish', 'English')
        with patch(TRANSLATE, side_effect=fake_translate), patch(
            'authoringtool.machine_translation.queue.apply_unit',
            side_effect=[RuntimeError('value too long for type character varying(255)')]
            + [None] * 50,
        ):
            drain_translation_queue()
        self.job.refresh_from_db()
        self.assertEqual(self.job.status, 'failed')
        self.assertIn('value too long', self.job.error)
        self.assertNotEqual(self.status(second), 'translating')

    def test_overlong_name_counts_as_failed_attempt(self):
        def long_helper(fields, *args):
            out = fake_translate(fields, *args)
            if 'helper' in out:
                out['helper'] = 'Σ' * 300 + ' $V=IR$'
            return out

        with patch(TRANSLATE, side_effect=long_helper):
            drain_translation_queue()
        self.job.refresh_from_db()
        self.assertEqual(self.job.status, 'failed')
        self.assertIn('too long', self.job.error)

    def test_very_long_source_name_can_be_copied(self):
        Scenario.objects.filter(pk=self.source.pk).update(name='N' * 255)
        self.source.refresh_from_db()
        job = copy_for_translation(self.source, self.owner, 'Portuguese', 'English')
        self.assertLessEqual(len(job.target_scenario.name), 255)
        self.assertTrue(job.target_scenario.name.endswith('(Portuguese)'))

    # I1 -------------------------------------------------------------------
    def test_claim_is_refreshed_after_each_unit(self):
        seen = []

        def record(fields, *args):
            self.job.refresh_from_db()
            seen.append(self.job.claimed_at)
            return fake_translate(fields, *args)

        start = timezone.now()
        times = iter(start + timedelta(seconds=10 * i) for i in range(1000))
        with patch(TRANSLATE, side_effect=record), \
                patch('authoringtool.machine_translation.queue.timezone.now', side_effect=lambda: next(times)):
            drain_translation_queue()
        self.assertEqual(self.status(), 'done')
        self.assertEqual(len(seen), len(set(seen)))  # a fresh claim time per unit

    def test_drain_stops_when_its_claim_was_taken_over(self):
        calls = []

        def taken_over(fields, *args):
            calls.append(1)
            ScenarioTranslationJob.objects.filter(pk=self.job.pk).update(
                claimed_at=timezone.now() + timedelta(days=1),
            )
            return fake_translate(fields, *args)

        with patch(TRANSLATE, side_effect=taken_over):
            drain_translation_queue()
        self.assertEqual(len(calls), 1)
        self.job.refresh_from_db()
        self.assertEqual(self.job.units_done, 1)

    def test_drain_does_not_release_another_runs_lock(self):
        def lock_taken(fields, *args):
            cache.set(LOCK_KEY, 'another-run', 60)
            return fake_translate(fields, *args)

        with patch(TRANSLATE, side_effect=lock_taken):
            drain_translation_queue()
        self.assertEqual(cache.get(LOCK_KEY), 'another-run')

    # I2 -------------------------------------------------------------------
    def test_unavailable_with_service_up_counts_as_unit_failure(self):
        with patch(TRANSLATE, side_effect=TranslationServiceUnavailable('HTTP 500')), \
                patch(CHECK, return_value=UP):
            drain_translation_queue()
        self.assertEqual(self.status(), 'failed')

    def test_unavailable_with_service_down_requeues(self):
        with patch(TRANSLATE, side_effect=TranslationServiceUnavailable('refused')), \
                patch(CHECK, return_value={'reachable': False, 'model_present': False}):
            drain_translation_queue()
        self.assertEqual(self.status(), 'waiting')


class EchoValidationTests(TestCase):
    # M3 (graded Important: untranslated text under a Machine-translated tag)
    def test_unchanged_sentence_is_rejected(self):
        text = '<p>What happens to the current if the resistance doubles?</p>'
        self.assertTrue(validate_unit({'text': text}, {'text': text}))

    def test_unchanged_short_name_is_accepted(self):
        self.assertEqual(validate_unit({'name': 'Ohm'}, {'name': 'Ohm'}), [])


class AnswerImageTests(TestCase):
    # I3
    def setUp(self):
        self.media = tempfile.mkdtemp()
        self.override = override_settings(MEDIA_ROOT=self.media)
        self.override.enable()
        self.owner = make_teacher('mt_fix_img_owner')
        self.source = build_source_scenario(self.owner)
        answer = Answer.objects.get(activity__scenario=self.source, text='It halves')
        answer.image.save('halves.png', ContentFile(b'png'), save=True)
        self.image_name = answer.image.name
        self.job = copy_for_translation(self.source, self.owner, 'Greek', 'English')
        ScenarioTranslationJob.objects.filter(pk=self.job.pk).update(status='done')
        self.client.force_login(self.owner)

    def tearDown(self):
        self.override.disable()
        shutil.rmtree(self.media, ignore_errors=True)

    def test_clearing_translation_image_keeps_source_file(self):
        ask = Activity.objects.get(scenario=self.job.target_scenario, name='Doubling resistance')
        copied = ask.answers.get(text='It halves')
        self.assertEqual(copied.image.name, self.image_name)
        self.client.post(
            reverse('updateAnswers', args=[ask.scenario_id, ask.phase_id, ask.id]),
            {'answers[0][id]': copied.id, 'answers[0][text]': 'Υποδιπλασιάζεται', 'clear_image_0': 'on'},
        )
        copied.refresh_from_db()
        self.assertFalse(copied.image)
        self.assertTrue(default_storage.exists(self.image_name))
