"""A proposal run reaches every confirmed translation.

Proposals are generated once, from the evidence pooled across the
translations. Each confirmed translation then gets the same proposals on its
own activities, in its own language, as its new current run. Each scenario's
teachers review their own copy.
"""
import json
from unittest.mock import patch

from django.utils import timezone

from authoringtool.models import (
    Activity,
    ActivityFlag,
    ActivityProposal,
    ProposalGenerationRun,
    ProposalStructuralFailure,
    Phase,
    Scenario,
    TranslationMatch,
)
from authoringtool.test_proposal_run_safety import GenerationJobFixture


def tag_language(data, language):
    """Stand-in translation: prefix every text with the language."""
    out = json.loads(json.dumps(data))
    if language and language.lower() not in ('en', 'english'):
        for field in ('activity_name', 'content', 'explanation'):
            out[field] = f'[{language}] {out[field]}'
    return out


class TranslationCopiesTests(GenerationJobFixture):
    def setUp(self):
        super().setUp()
        self.scenario.language = 'en'
        self.scenario.save(update_fields=['language'])
        self.greek, self.greek_activities = self.make_translation('Εκκρεμές', 'el')

    def make_translation(self, name, language, mapped=2):
        scenario = Scenario.objects.create(
            name=name, language=language,
            created_by=self.owner, updated_by=self.owner,
        )
        phase = Phase.objects.create(
            name='Φ', scenario=scenario, created_by=self.owner, updated_by=self.owner,
        )
        activities = [
            Activity.objects.create(
                name=f'{name} Q{number}', text='q', plain_text='q', phase=phase,
                scenario=scenario, activity_type=self.activities[0].activity_type,
                created_by=self.owner, updated_by=self.owner,
            )
            for number in (1, 2)
        ]
        TranslationMatch.objects.create(
            scenario_a=self.scenario, scenario_b=scenario,
            confidence=100, exact_match=True, status='confirmed',
            activity_map=[
                [ours.id, theirs.id]
                for ours, theirs in list(zip(self.activities, activities))[:mapped]
            ],
        )
        return scenario, activities

    def current(self, scenario):
        return ProposalGenerationRun.objects.filter(scenario=scenario, is_current=True).first()

    def test_translation_gets_the_same_proposals_in_its_language(self):
        self.flag(self.activities[0])
        self.run_job(proposals=[self.written()], translate=tag_language)

        source = self.current(self.scenario).proposals.get()
        copy = self.current(self.greek).proposals.get()
        self.assertEqual(copy.activity, self.greek_activities[0])
        self.assertEqual(copy.phase, self.greek_activities[0].phase)
        self.assertEqual(copy.proposal_type, source.proposal_type)
        self.assertEqual(copy.status, 'new')
        self.assertEqual(copy.json_action, source.json_action)
        translated = json.loads(copy.json_translated_action)
        self.assertEqual(translated['content'], '[el] x')
        self.assertEqual(translated['target_category'], ['Low'])
        self.assertEqual(copy.translated_action, copy.json_translated_action)
        # Same decision, so the same provenance.
        self.assertEqual(copy.selection_policy, source.selection_policy)
        self.assertEqual(copy.action_probabilities, source.action_probabilities)
        self.assertEqual(copy.prompt_version, source.prompt_version)
        # The flags and risk categories come along, on the translation's activity.
        flag = copy.flag.get()
        self.assertEqual(flag.activity, self.greek_activities[0])
        self.assertEqual(flag.scenario, self.greek)
        self.assertEqual(flag.flag_type, 'Systemic failure')
        self.assertEqual(list(copy.categories_in_risk.values_list('name', flat=True)), ['Low'])

    def test_copy_run_says_where_it_came_from(self):
        self.flag(self.activities[0])
        self.run_job(proposals=[self.written()], translate=tag_language)

        copied = self.current(self.greek).evidence_summary['copied_from']
        self.assertEqual(copied['scenario_id'], self.scenario.id)
        self.assertEqual(copied['run_id'], self.current(self.scenario).id)

    def test_translation_old_run_moves_to_history(self):
        old = ProposalGenerationRun.objects.create(scenario=self.greek, is_current=True)
        self.flag(self.activities[0])
        self.run_job(proposals=[self.written()], translate=tag_language)

        old.refresh_from_db()
        self.assertFalse(old.is_current)
        self.assertNotEqual(self.current(self.greek), old)

    def test_reviewing_one_copy_leaves_the_other_untouched(self):
        self.flag(self.activities[0])
        self.run_job(proposals=[self.written()], translate=tag_language)

        copy = self.current(self.greek).proposals.get()
        copy.status = 'accepted'
        copy.save()
        self.assertEqual(self.current(self.scenario).proposals.get().status, 'new')

    def test_activity_without_a_counterpart_is_not_copied(self):
        other, other_activities = self.make_translation('Pendule', 'fr', mapped=1)
        for activity in self.activities:
            self.flag(activity)
        self.run_job(proposals=[self.written(), self.written()], translate=tag_language)

        self.assertEqual(self.current(self.greek).proposals.count(), 2)
        copy = self.current(other).proposals.get()
        self.assertEqual(copy.activity, other_activities[0])

    def test_archived_translation_is_left_alone(self):
        self.greek.archived_at = timezone.now()
        self.greek.save(update_fields=['archived_at'])
        self.flag(self.activities[0])
        self.run_job(proposals=[self.written()], translate=tag_language)

        self.assertFalse(ProposalGenerationRun.objects.filter(scenario=self.greek).exists())

    def test_translation_with_its_own_job_running_is_left_alone(self):
        old = ProposalGenerationRun.objects.create(scenario=self.greek, is_current=True)
        self.flag(self.activities[0])

        def status(scenario_id):
            return {'running': scenario_id == self.greek.id}

        with patch('authoringtool.tasks.job_status', side_effect=status):
            self.run_job(proposals=[self.written()], translate=tag_language)

        self.assertEqual(self.current(self.greek), old)

    def test_failed_translation_keeps_english_and_is_recorded(self):
        self.flag(self.activities[0])

        def translate(data, language):
            if language == 'el':
                raise RuntimeError('translation server down')
            return dict(data)

        self.run_job(proposals=[self.written()], translate=translate)

        copy = self.current(self.greek).proposals.get()
        self.assertEqual(json.loads(copy.json_translated_action)['content'], 'x')
        failure = ProposalStructuralFailure.objects.get(scenario=self.greek)
        self.assertEqual(failure.stage, 'translation')

    def test_aborted_run_copies_nothing(self):
        from authoringtool.tasks import ProposalRunAborted, ProposalValidationError
        self.flag(self.activities[0])
        with self.assertRaises(ProposalRunAborted):
            self.run_job(proposals=[ProposalValidationError(['bad'])], translate=tag_language)

        self.assertFalse(ProposalGenerationRun.objects.filter(scenario=self.greek).exists())

    def test_run_without_risks_also_clears_the_translations(self):
        old = ProposalGenerationRun.objects.create(scenario=self.greek, is_current=True)
        self.run_job(translate=tag_language)

        self.assertNotEqual(self.current(self.greek), old)
        self.assertEqual(self.current(self.greek).proposals.count(), 0)

    def test_finish_message_names_the_translations(self):
        self.flag(self.activities[0])
        with patch('authoringtool.tasks.JobProgress.finish') as finish:
            self.run_job(proposals=[self.written()], translate=tag_language)
        message = finish.call_args.args[1]
        self.assertIn('Εκκρεμές', message)
