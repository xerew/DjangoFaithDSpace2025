"""A proposal run never hides the current proposals unless it produced new ones.

Before the job touches anything it asks the proposal model for a one-word
answer. The new run is built out of sight and replaces the current one only
when it finished; if the AI server stops answering, the current proposals stay.
"""
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import requests
from django.conf import settings
from django.contrib.auth.models import Group, User
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from authoringtool.evidence import get_evidence_context
from authoringtool.machine_translation.health import (
    AIServerUnavailable,
    probe_proposal_model,
    proposal_server_status,
)
from authoringtool.models import (
    Activity,
    ActivityFlag,
    ActivityProposal,
    ActivityType,
    Phase,
    ProposalGenerationRun,
    Scenario,
)
from authoringtool.tasks import (
    ProposalRunAborted,
    ProposalValidationError,
    generate_llm_context_for_scenario,
    request_validated_proposal,
)

LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}
HEALTH_POST = 'authoringtool.machine_translation.health.requests.post'


def http_response(status=200, reason='OK', body=None):
    response = MagicMock(status_code=status, reason=reason)
    response.json.return_value = body if body is not None else {'response': 'OK'}
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f'{status} Server Error: {reason} for url: http://ollama-tunnel:11434/api/generate',
            response=response,
        )
    return response


@override_settings(CACHES=LOCMEM)
class ProbeProposalModelTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_passes_when_the_model_answers(self):
        with patch(HEALTH_POST, return_value=http_response()) as post:
            probe_proposal_model()

        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['model'], settings.LLM_PROPOSAL_MODEL)
        self.assertEqual(payload['options']['num_predict'], 1)
        self.assertTrue(post.call_args.kwargs['timeout'])

    def test_bad_gateway_is_reported_and_marks_the_server_offline(self):
        with patch(HEALTH_POST, return_value=http_response(502, 'Bad Gateway')):
            with self.assertRaises(AIServerUnavailable) as raised:
                probe_proposal_model()

        self.assertIn('502 Bad Gateway', str(raised.exception))
        self.assertEqual(proposal_server_status()['state'], 'offline')

    def test_connection_failure_is_reported(self):
        with patch(HEALTH_POST, side_effect=requests.ConnectionError('refused')):
            with self.assertRaises(AIServerUnavailable) as raised:
                probe_proposal_model()

        self.assertIn('could not connect', str(raised.exception))

    def test_ollama_error_message_is_reported(self):
        body = {'error': 'model "qwen" not found'}
        with patch(HEALTH_POST, return_value=http_response(body=body)):
            with self.assertRaises(AIServerUnavailable) as raised:
                probe_proposal_model()

        self.assertIn('not found', str(raised.exception))


class RequestValidatedProposalServerFailureTests(SimpleTestCase):
    def test_server_failure_on_every_attempt_is_not_a_bad_proposal(self):
        with patch('authoringtool.tasks.requests.post',
                   return_value=http_response(502, 'Bad Gateway')):
            with self.assertRaises(AIServerUnavailable):
                request_validated_proposal('prompt', required_action='skip')

    def test_invalid_answers_are_still_a_bad_proposal(self):
        with patch('authoringtool.tasks.requests.post',
                   return_value=http_response(body={'response': 'not json'})):
            with self.assertRaises(ProposalValidationError) as raised:
                request_validated_proposal('prompt', required_action='skip')

        self.assertNotIsInstance(raised.exception, AIServerUnavailable)


class RunSwitchTests(TestCase):
    def setUp(self):
        owner = User.objects.create_user('switch_owner')
        self.scenario = Scenario.objects.create(name='S', created_by=owner, updated_by=owner)
        self.owner = owner

    def test_hidden_run_leaves_the_current_one_current(self):
        old = ProposalGenerationRun.objects.create(scenario=self.scenario, is_current=True)

        new = ProposalGenerationRun.start_new(self.scenario, self.owner, current=False)

        old.refresh_from_db()
        self.assertTrue(old.is_current)
        self.assertFalse(new.is_current)

    def test_make_current_moves_the_old_run_to_history(self):
        old = ProposalGenerationRun.objects.create(scenario=self.scenario, is_current=True)
        new = ProposalGenerationRun.start_new(self.scenario, self.owner, current=False)

        new.make_current()

        old.refresh_from_db()
        new.refresh_from_db()
        self.assertFalse(old.is_current)
        self.assertTrue(new.is_current)


def proposal_json(action='create'):
    return {
        'action': action, 'activity_name': 'New', 'activity_type': 'Explanation',
        'content': 'x', 'answers': [], 'explanation': 'e', 'insert_location': 'after',
    }


class GenerationJobFixture(TestCase):
    """The whole job with the AI server, RAG and risk detection replaced."""

    def setUp(self):
        self.owner = User.objects.create_user('job_owner')
        self.scenario = Scenario.objects.create(
            name='Pendulum', created_by=self.owner, updated_by=self.owner,
            llm_context='scenario context',
        )
        phase = Phase.objects.create(
            name='P', scenario=self.scenario, created_by=self.owner,
            updated_by=self.owner, llm_context='phase context',
        )
        question = ActivityType.objects.create(
            name='Question', created_by=self.owner, updated_by=self.owner,
        )
        self.activities = [
            Activity.objects.create(
                name=f'Q{number}', text='q', plain_text='q', phase=phase,
                scenario=self.scenario, activity_type=question,
                llm_context='ctx', short_llm_summary='sum',
                related_act_llm_context={'High': 'a', 'Moderate': 'b', 'Low': 'c'},
                created_by=self.owner, updated_by=self.owner,
            )
            for number in (1, 2)
        ]
        self.old_run = ProposalGenerationRun.objects.create(
            scenario=self.scenario, is_current=True,
        )
        ActivityProposal.objects.create(
            scenario=self.scenario, generation_run=self.old_run, phase=phase,
            activity=self.activities[0], proposal_type='skip', suggested_action='{}',
            translated_action='{}', json_action='{}', json_translated_action='{}',
        )

    def flag(self, activity):
        ActivityFlag.objects.create(
            activity=activity, scenario=self.scenario, phase=activity.phase,
            category='Low', flag_type='Systemic failure', flag_reason='r',
            evidence_scope='local',
            evidence_signature=get_evidence_context(self.scenario, scope='local')['signature'],
        )

    def run_job(self, proposals=None, probe=None, force=False, translate=None):
        replacements = {
            'probe_proposal_model': MagicMock(side_effect=probe),
            'chunk_and_index_pdfs': MagicMock(),
            'ollama_text': MagicMock(return_value='ctx'),
            'calculate_activities_in_risk': MagicMock(),
            'request_validated_proposal': MagicMock(side_effect=proposals or []),
            'translate_proposal_fields': MagicMock(
                side_effect=translate or (lambda data, language: dict(data)),
            ),
            'validate_proposal_data': MagicMock(side_effect=lambda data, **kwargs: data),
            'get_data_insight': MagicMock(return_value=('', '')),
            'get_phase_based_prior_summary': MagicMock(return_value=''),
        }
        patchers = [patch(f'authoringtool.tasks.{name}', value)
                    for name, value in replacements.items()]
        patchers.append(patch.object(pd.DataFrame, 'to_csv'))
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        return generate_llm_context_for_scenario(
            self.scenario.id, force_rebuild=force, triggered_by_id=self.owner.id,
        )

    def written(self):
        structured = proposal_json()
        return (json.dumps(structured), structured)

    def current_run(self):
        return ProposalGenerationRun.objects.get(scenario=self.scenario, is_current=True)


class GenerationJobTests(GenerationJobFixture):
    def test_server_down_at_start_changes_nothing(self):
        self.flag(self.activities[0])

        with self.assertRaises(ProposalRunAborted) as raised:
            self.run_job(probe=AIServerUnavailable("The AI server isn't answering (502 Bad Gateway)."),
                         force=True)

        self.assertIn('502 Bad Gateway', str(raised.exception))
        self.assertIn('current proposals were kept', str(raised.exception))
        self.assertEqual(self.current_run(), self.old_run)
        self.assertEqual(ProposalGenerationRun.objects.count(), 1)
        self.scenario.refresh_from_db()
        self.assertEqual(self.scenario.llm_context, 'scenario context')

    def test_server_lost_during_the_run_keeps_the_current_proposals(self):
        for activity in self.activities:
            self.flag(activity)

        with self.assertRaises(ProposalRunAborted) as raised:
            self.run_job(proposals=[
                self.written(),
                AIServerUnavailable('502 Bad Gateway'),
            ])

        self.assertIn('stopped answering', str(raised.exception))
        self.assertEqual(self.current_run(), self.old_run)
        unfinished = ProposalGenerationRun.objects.exclude(pk=self.old_run.pk).get()
        self.assertFalse(unfinished.is_current)

    def test_finished_run_replaces_the_current_one(self):
        for activity in self.activities:
            self.flag(activity)

        self.run_job(proposals=[self.written(), self.written()])

        new_run = self.current_run()
        self.assertNotEqual(new_run, self.old_run)
        self.assertEqual(new_run.proposals.count(), 2)

    def test_run_where_no_proposal_could_be_written_keeps_the_current_one(self):
        self.flag(self.activities[0])

        with self.assertRaises(ProposalRunAborted) as raised:
            self.run_job(proposals=[ProposalValidationError(['bad json'])])

        self.assertIn('No proposal could be written', str(raised.exception))
        self.assertEqual(self.current_run(), self.old_run)

    def test_run_without_risks_replaces_the_current_one(self):
        self.run_job()

        new_run = self.current_run()
        self.assertNotEqual(new_run, self.old_run)
        self.assertEqual(new_run.proposals.count(), 0)


class TaskStatusTests(TestCase):
    def setUp(self):
        teachers, _ = Group.objects.get_or_create(name='teachers')
        user = User.objects.create_user('status_teacher')
        user.groups.add(teachers)
        self.client.force_login(user)

    def test_failed_job_reports_its_reason(self):
        failed = MagicMock(state='FAILURE', result=ProposalRunAborted('The AI server stopped answering.'))
        failed.failed.return_value = True
        with patch('authoringtool.views.AsyncResult', return_value=failed):
            response = self.client.get(reverse('llm_context_status', args=['task-1']))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            'state': 'FAILURE', 'result': 'The AI server stopped answering.',
        })
