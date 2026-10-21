from django.conf import settings
from django.contrib.auth.models import User
from django.db import IntegrityError, transaction
from django.test import TestCase

from authoringtool.models import (
    Scenario,
    ScenarioCollaborator,
    ScenarioTranslationJob,
    TranslationMatch,
)


def make_scenario(user, name, language='English'):
    return Scenario.objects.create(
        name=name, language=language, created_by=user, updated_by=user,
    )


class MachineTranslationModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('mt_owner', password='x')
        self.source = make_scenario(self.user, 'Ohm')
        self.target = make_scenario(self.user, 'Ohm (Greek)', 'Greek')

    def test_job_defaults(self):
        job = ScenarioTranslationJob.objects.create(
            source_scenario=self.source,
            target_scenario=self.target,
            source_language='English',
            target_language='Greek',
            requested_by=self.user,
        )
        self.assertEqual(job.status, 'waiting')
        self.assertEqual(job.units_done, 0)
        self.assertEqual(job.id_map, {})

    def test_collaborator_unique_per_scenario(self):
        ScenarioCollaborator.objects.create(
            scenario=self.target, user=self.user, role='translation_reviewer',
        )
        with self.assertRaises(IntegrityError), transaction.atomic():
            ScenarioCollaborator.objects.create(
                scenario=self.target, user=self.user, role='co_author',
            )

    def test_translation_match_origin_defaults_to_scan(self):
        match = TranslationMatch.objects.create(
            scenario_a=self.source, scenario_b=self.target,
        )
        self.assertEqual(match.origin, 'scan')

    def test_machine_translation_model_setting_default(self):
        self.assertEqual(settings.LLM_MACHINE_TRANSLATION_MODEL, 'gemma4:31b')
