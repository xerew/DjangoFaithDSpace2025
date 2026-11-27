"""Proposals must make sense for the evidence behind them.

- Skip is offered only when every flag says the group finds the activity
  easy (too fast, answered everything right) or the activity is disconnected.
  A group that is struggling never gets "skip this activity".
- Proposals produce text only, so the hints and the prompt never ask the
  model for videos, animations or images it cannot make.
"""
import re
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase

from authoringtool import tasks
from authoringtool.models import BanditPolicyConfiguration, QValue
from authoringtool.tasks import (
    allowed_proposal_actions,
    choose_proposal_action,
    get_data_insight,
)
from authoringtool.test_proposal_run_safety import GenerationJobFixture


def flag(flag_type, category='Low', reason='', isolated=False):
    return SimpleNamespace(
        flag_type=flag_type, category=category, flag_reason=reason,
        is_isolated=isolated, activity=SimpleNamespace(name='A'),
    )


STRUGGLING = flag(
    'Dynamic correctness threshold',
    reason="Dynamic correctness: Low group's %Wrong (92.0%) exceeds mean+std threshold",
)
TOO_FAST = flag(
    'Engagement risk (too fast, question)',
    reason="Engagement risk: Low group's Avg Time (3.0s) is below mean-std",
)
ALL_WRONG_SLOW = flag(
    'Timing discrepancy under extreme correctness',
    reason='Low group answered everything incorrectly (100% wrong) yet spent 80s',
)
ALL_RIGHT_SLOW = flag(
    'Timing discrepancy under extreme correctness',
    reason='Low group answered everything correctly (0% wrong) yet spent 80s',
)


class AllowedActionsTests(TestCase):
    def test_struggling_group_cannot_get_skip(self):
        for struggling in (
            STRUGGLING,
            flag('Systemic failure', category='All'),
            flag('Extreme correctness pattern', category='Moderate'),
            flag('Engagement risk (too slow, question)'),
            ALL_WRONG_SLOW,
        ):
            with self.subTest(struggling.flag_type):
                self.assertEqual(
                    allowed_proposal_actions([struggling]), ('create', 'revise'),
                )

    def test_easy_or_disconnected_activity_may_be_skipped(self):
        for easy in (TOO_FAST, ALL_RIGHT_SLOW, flag('Anything', isolated=True)):
            with self.subTest(easy.flag_type):
                self.assertIn('skip', allowed_proposal_actions([easy]))

    def test_one_struggling_flag_rules_out_skip(self):
        self.assertNotIn('skip', allowed_proposal_actions([TOO_FAST, STRUGGLING]))

    def test_no_flags_keeps_every_action(self):
        self.assertEqual(set(allowed_proposal_actions([])), {'create', 'revise', 'skip'})


class ChooseActionRespectsEvidenceTests(TestCase):
    def test_cold_start_never_skips_a_struggling_group(self):
        decision = choose_proposal_action([STRUGGLING])
        self.assertEqual(decision.policy, 'weighted_exploration')
        self.assertNotIn('skip', decision.probabilities)
        self.assertAlmostEqual(decision.probabilities['create'], 0.5 / 0.7)
        self.assertAlmostEqual(decision.probabilities['revise'], 0.2 / 0.7)
        # A roll that lands in skip's share of the full weights must not pick it.
        with patch('authoringtool.tasks.random.random', return_value=0.99):
            for _ in range(5):
                self.assertNotEqual(choose_proposal_action([STRUGGLING]).action, 'skip')

    def test_cold_start_still_offers_skip_when_the_activity_is_easy(self):
        decision = choose_proposal_action([TOO_FAST])
        self.assertAlmostEqual(decision.probabilities['skip'], 0.3)

    def _mature(self, flag_obj, rewards):
        configuration = BanditPolicyConfiguration.get_active()
        configuration.minimum_context_rewards = 1
        return configuration, [
            QValue.objects.create(
                flag_type=flag_obj.flag_type, category=flag_obj.category,
                action=action, reward_count=10, reward_sum=rewards[action],
                positive_reward_count=int(rewards[action]),
                negative_reward_count=10 - int(rewards[action]),
            )
            for action in ('create', 'revise', 'skip')
        ]

    def test_ucb_never_skips_a_struggling_group_even_when_skip_scored_best(self):
        configuration, _ = self._mature(STRUGGLING, {'create': 1, 'revise': 1, 'skip': 9})
        configuration.policy = 'ucb'
        configuration.save()
        decision = choose_proposal_action([STRUGGLING])
        self.assertEqual(decision.policy, 'ucb')
        self.assertNotEqual(decision.action, 'skip')
        self.assertEqual(decision.probability, 1.0)

    def test_thompson_never_skips_a_struggling_group(self):
        configuration, _ = self._mature(STRUGGLING, {'create': 1, 'revise': 1, 'skip': 9})
        configuration.policy = 'thompson'
        configuration.save()
        for _ in range(10):
            decision = choose_proposal_action([STRUGGLING])
            self.assertEqual(decision.policy, 'thompson')
            self.assertNotEqual(decision.action, 'skip')
            self.assertNotIn('skip', decision.probabilities)
            self.assertAlmostEqual(sum(decision.probabilities.values()), 1.0, places=6)


MEDIA = re.compile(r'video|animation|image|multimedia|slide', re.IGNORECASE)


class HintsAskOnlyForTextTests(TestCase):
    def test_no_hint_asks_for_media_the_platform_cannot_make(self):
        for flag_obj in (
            STRUGGLING,
            flag('Dynamic correctness threshold',
                 reason="Dynamic correctness: Low group's %Wrong (30.0%) exceeds"),
            flag('Dynamic correctness threshold',
                 reason="Dynamic correctness: Low group's %Wrong (55.0%) exceeds"),
            flag('Engagement risk (too slow, question)',
                 reason="Engagement risk: Low group's Avg Time (90.0s) is above"),
            flag('Engagement risk (too fast, question)',
                 reason="Engagement risk: Low group's Avg Time (2.0s) is below"),
            flag('Extreme correctness pattern', category='Moderate'),
            ALL_WRONG_SLOW,
            ALL_RIGHT_SLOW,
            flag('Timing discrepancy under extreme correctness', reason='other'),
            flag('Systemic failure', category='All'),
            flag('Anything', isolated=True),
        ):
            with self.subTest(flag_obj.flag_type, reason=flag_obj.flag_reason):
                insight, resolution = get_data_insight(flag_obj)
                self.assertIsNone(MEDIA.search(resolution), resolution)

    def test_struggling_hints_do_not_suggest_skipping(self):
        for flag_obj in (STRUGGLING, ALL_WRONG_SLOW, flag('Systemic failure', category='All')):
            _, resolution = get_data_insight(flag_obj)
            self.assertNotIn('skip', resolution.lower())


class PromptForbidsMissingMediaTests(GenerationJobFixture):
    def test_prompt_tells_the_model_not_to_refer_to_media(self):
        self.flag(self.activities[0])
        self.run_job(proposals=[self.written()])
        prompt = tasks.request_validated_proposal.call_args.args[0]
        self.assertIn('never mention or rely on a video', prompt.lower())
