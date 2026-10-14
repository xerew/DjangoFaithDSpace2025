"""Proposals record how they were produced (M8): the LLM, the prompt version,
the bandit policy and the probability it gave the chosen action. Without the
probability, logged proposals cannot be used for off-policy evaluation."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import TestCase, override_settings

from authoringtool.models import (
    Activity,
    ActivityProposal,
    ActivityType,
    BanditPolicyConfiguration,
    Phase,
    QValue,
    Scenario,
)
from authoringtool.tasks import (
    PROPOSAL_PROMPT_VERSION,
    choose_proposal_action,
    proposal_provenance,
    request_validated_proposal,
)

FLAG = SimpleNamespace(flag_type='Systemic failure', category='Low')


def mature_context(**counts):
    for action in ('create', 'revise', 'skip'):
        QValue.objects.create(
            flag_type=FLAG.flag_type, category=FLAG.category, action=action,
            reward_count=10, reward_sum=counts.get(action, 0),
            positive_reward_count=5, negative_reward_count=5,
        )


class ChooseProposalActionTests(TestCase):
    def test_cold_start_logs_configured_weights(self):
        decision = choose_proposal_action([])
        self.assertEqual(decision.policy, 'weighted_exploration')
        self.assertAlmostEqual(decision.probabilities['create'], 0.5)
        self.assertAlmostEqual(decision.probabilities['skip'], 0.3)
        self.assertAlmostEqual(decision.probabilities['revise'], 0.2)
        self.assertEqual(decision.probability, decision.probabilities[decision.action])

    def test_ucb_is_deterministic_once_every_action_was_tried(self):
        configuration = BanditPolicyConfiguration.get_active()
        configuration.policy = 'ucb'
        configuration.minimum_context_rewards = 1
        configuration.save()
        mature_context(create=9.0, revise=1.0, skip=1.0)
        decision = choose_proposal_action([FLAG])
        self.assertEqual(decision.policy, 'ucb')
        self.assertEqual(decision.action, 'create')
        self.assertEqual(decision.probability, 1.0)

    def test_thompson_logs_an_estimated_probability(self):
        configuration = BanditPolicyConfiguration.get_active()
        configuration.policy = 'thompson'
        configuration.minimum_context_rewards = 1
        configuration.save()
        mature_context()
        with patch('authoringtool.tasks.random.betavariate', side_effect=[0.9, 0.2, 0.1]):
            decision = choose_proposal_action([FLAG])
        self.assertEqual(decision.policy, 'thompson')
        self.assertEqual(decision.action, 'create')
        self.assertAlmostEqual(sum(decision.probabilities.values()), 1.0, places=6)
        self.assertGreater(decision.probability, 0)
        self.assertEqual(decision.contexts, [['Systemic failure', 'Low']])


class ProposalProvenanceTests(TestCase):
    def test_proposal_stores_model_prompt_policy_and_probability(self):
        owner = User.objects.create_user('prov_owner')
        scenario = Scenario.objects.create(name='Prov', created_by=owner, updated_by=owner)
        phase = Phase.objects.create(name='P', scenario=scenario, created_by=owner, updated_by=owner)
        activity_type = ActivityType.objects.create(name='Question', created_by=owner, updated_by=owner)
        activity = Activity.objects.create(
            name='Q', text='q', phase=phase, scenario=scenario, activity_type=activity_type,
            created_by=owner, updated_by=owner,
        )
        decision = choose_proposal_action([])
        with override_settings(LLM_PROPOSAL_MODEL='test-model:1b'):
            proposal = ActivityProposal.objects.create(
                scenario=scenario, phase=phase, activity=activity,
                proposal_type=decision.action, suggested_action='{}',
                translated_action='{}', json_action='{}', json_translated_action='{}',
                **proposal_provenance(decision),
            )
        proposal.refresh_from_db()
        self.assertEqual(proposal.llm_model, 'test-model:1b')
        self.assertEqual(proposal.prompt_version, PROPOSAL_PROMPT_VERSION)
        self.assertEqual(proposal.selection_policy, 'weighted_exploration')
        self.assertAlmostEqual(proposal.selection_probability, decision.probability)
        self.assertEqual(set(proposal.action_probabilities), {'create', 'skip', 'revise'})


class ProposalModelSettingTests(TestCase):
    @override_settings(LLM_PROPOSAL_MODEL='test-model:1b')
    def test_request_uses_the_configured_model(self):
        response = MagicMock()
        response.json.return_value = {'response': json.dumps({
            'action': 'skip', 'activity_name': 'A', 'activity_type': 'Explanation',
            'content': 'x', 'answers': [], 'explanation': 'e',
        })}
        with patch('authoringtool.tasks.requests.post', return_value=response) as post, \
                patch('authoringtool.tasks.validate_proposal_data', side_effect=lambda data, **kw: data):
            request_validated_proposal('prompt', required_action='skip')
        self.assertEqual(post.call_args.kwargs['json']['model'], 'test-model:1b')
